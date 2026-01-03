#!/usr/bin/env python3
"""
Iron Condor Screener & Analyzer
Finds the best iron condor opportunities with risk analysis.
"""

import os
import math
import argparse
import pandas as pd
import numpy as np
import json
import concurrent.futures
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from massive import RESTClient
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import date, timedelta


# Configuration
ET = ZoneInfo("America/New_York")
load_dotenv()

MAX_OPTIONS_PER_SIDE = 30
STRIKE_DISTANCE_PCT = 0.2
MAX_SPREAD_PCT = 0.08
MAX_SPREAD_ABS = 50.0
MIN_SPREAD_ABS = 0.5


def parse_range(value: Optional[str]) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Parse CLI range strings formatted as 'min,max'."""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Range must be formatted as min,max")
    try:
        low = float(parts[0]) if parts[0] else None
        high = float(parts[1]) if parts[1] else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Range values must be numeric") from exc
    if low is not None and high is not None and low > high:
        raise argparse.ArgumentTypeError("Range min cannot exceed max")
    return (low, high)


def value_in_range(value: Optional[float], bounds: Optional[Tuple[Optional[float], Optional[float]]]) -> bool:
    """Utility to determine whether a value satisfies optional range bounds."""
    if bounds is None:
        return True
    if value is None:
        return False
    low, high = bounds
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True

@dataclass
class IronCondor:
    """Represents an iron condor strategy"""
    expiration: str
    call_spread: Tuple[float, float]  # (sell_strike, buy_strike)
    put_spread: Tuple[float, float]   # (sell_strike, buy_strike)
    net_credit: float
    max_profit: float
    max_loss: float
    profit_zone: Tuple[float, float]  # (lower_bound, upper_bound)
    probability_of_profit: float
    risk_reward_ratio: float
    days_to_expiration: int
    spot_price: float
    credit_ratio: float
    call_sell_delta: Optional[float] = None
    call_sell_theta: Optional[float] = None
    call_sell_iv: Optional[float] = None
    call_sell_volume: Optional[int] = None
    call_sell_open_interest: Optional[int] = None
    call_buy_delta: Optional[float] = None
    call_buy_theta: Optional[float] = None
    call_buy_iv: Optional[float] = None
    call_buy_volume: Optional[int] = None
    call_buy_open_interest: Optional[int] = None
    put_sell_delta: Optional[float] = None
    put_sell_theta: Optional[float] = None
    put_sell_iv: Optional[float] = None
    put_sell_volume: Optional[int] = None
    put_sell_open_interest: Optional[int] = None
    put_buy_delta: Optional[float] = None
    put_buy_theta: Optional[float] = None
    put_buy_iv: Optional[float] = None
    put_buy_volume: Optional[int] = None
    put_buy_open_interest: Optional[int] = None

class IronCondorScreener:
    def __init__(self, client: Optional[RESTClient] = None, api_key: Optional[str] = None):
        """Initialize the screener with Massive API client"""
        if client is not None:
            self.client = client
        else:
            api_key = api_key or os.getenv('MASSIVE_API_KEY')
            if not api_key:
                raise ValueError("MASSIVE_API_KEY not found in environment variables")
            self.client = RESTClient(api_key=api_key)
        
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.free_tier = os.getenv('FREETIER', 'false').lower() == 'true'
        
    def log(self, message: str):
        """Log message if debug is enabled"""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get current stock price"""
        try:
            last_trade = self.client.get_last_trade(symbol)
            return float(last_trade.price)
        except Exception as e:
            self.log(f"Error getting current price for {symbol}: {e}")
            return 0.0
        
    def get_last_close(self, symbol: str) -> float:
        # previous trading day – for demo just "yesterday"
        end = date.today() - timedelta(days=7)
        start = end

        try:
            aggs = list(self.client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start.isoformat(),
                to=end.isoformat(),
                limit=1,
            ))
            if not aggs:
                raise RuntimeError("No aggregates returned")
            return float(aggs[-1].close)
        except Exception as e:
            self.log(f"Error getting last close for {symbol}: {e}")
            return 0.0


    
    def check_upcoming_earnings(self, symbol: str, max_days: int = 30) -> bool:
        """Check if there are upcoming earnings within max_days"""
        try:
            today = datetime.now(ET).date()
            end_date = today + timedelta(days=max_days)
            
            # Get earnings data from Benzinga
            earnings = list(self.client.list_benzinga_earnings(
                ticker=symbol,
                date_gte=today.strftime('%Y-%m-%d'),
                date_lte=end_date.strftime('%Y-%m-%d'),
                limit=10
            ))
            
            has_earnings = len(earnings) > 0
            if has_earnings:
                self.log(f"Found {len(earnings)} upcoming earnings for {symbol}")
            else:
                self.log(f"No upcoming earnings found for {symbol} in next {max_days} days")
            
            return has_earnings
            
        except Exception as e:
            self.log(f"Error checking earnings for {symbol}: {e}")
            return False
    
    def _get_relevant_expirations(self, symbol: str, max_days: int) -> List[str]:
        """Return sorted expirations within the max_days window."""
        today = datetime.now(ET).date()
        end_date = today + timedelta(days=max_days)
        expirations: set[str] = set()
        
        try:
            for contract in self.client.list_options_contracts(
                underlying_ticker=symbol,
                expired=False,
                limit=1000
            ):
                exp_date_str = contract.expiration_date
                if not exp_date_str:
                    continue
                
                try:
                    exp_date = datetime.fromisoformat(str(exp_date_str)).date()
                except ValueError:
                    continue
                
                if today <= exp_date <= end_date:
                    expirations.add(str(exp_date))
        except Exception as e:
            self.log(f"Error fetching expirations for {symbol}: {e}")
        
        sorted_expirations = sorted(expirations)
        self.log(f"Found {len(sorted_expirations)} expirations for {symbol}: {sorted_expirations}")
        return sorted_expirations
    
    def _extract_implied_volatility(self, option) -> Optional[float]:
        """Safely extract implied volatility from a snapshot."""
        possible_iv = getattr(option, 'implied_volatility', None)
        if possible_iv:
            try:
                return float(possible_iv)
            except (TypeError, ValueError):
                pass
        
        greeks = getattr(option, 'greeks', None)
        if greeks:
            greeks_iv = getattr(greeks, 'implied_volatility', None)
            if greeks_iv:
                try:
                    return float(greeks_iv)
                except (TypeError, ValueError):
                    return None
        return None
    
    def _fetch_options_for_expiration(self, symbol: str, expiration: str) -> Tuple[str, Dict[str, List[Dict]]]:
        """Fetch option snapshots for a single expiration."""
        options_bucket = {'calls': [], 'puts': []}
        params = {'expiration_date': expiration}
        
        try:
            for option in self.client.list_snapshot_options_chain(
                underlying_asset=symbol,
                params=params
            ):
                exp_date = option.details.expiration_date
                if not exp_date or str(exp_date) != expiration:
                    continue
                
                delta = None
                theta = None
                greeks = getattr(option, 'greeks', None)
                if greeks:
                    try:
                        delta = float(getattr(greeks, 'delta', None))
                    except (TypeError, ValueError):
                        delta = None
                    try:
                        theta = float(getattr(greeks, 'theta', None))
                    except (TypeError, ValueError):
                        theta = None

                option_data = {
                    'strike': float(option.details.strike_price) if option.details.strike_price else 0.0,
                    'bid': float(option.last_quote.bid) if option.last_quote and option.last_quote.bid else 0.0,
                    'ask': float(option.last_quote.ask) if option.last_quote and option.last_quote.ask else 0.0,
                    'volume': int(option.day.volume) if option.day and option.day.volume else 0,
                    'open_interest': int(option.open_interest) if option.open_interest else 0,
                    'implied_volatility': self._extract_implied_volatility(option),
                    'delta': delta,
                    'theta': theta,
                }
                
                if option.details.contract_type == "call":
                    options_bucket['calls'].append(option_data)
                elif option.details.contract_type == "put":
                    options_bucket['puts'].append(option_data)
        except Exception as e:
            self.log(f"Error fetching snapshots for {symbol} {expiration}: {e}")
        
        return expiration, options_bucket
    
    def fetch_and_group_options(self, symbol: str, max_days: int = 30) -> Dict[str, Dict]:
        """Fetch option data grouped by expiration with concurrency and filtering."""
        expirations = self._get_relevant_expirations(symbol, max_days)
        if not expirations:
            return {}
        
        grouped_options: Dict[str, Dict[str, List[Dict]]] = {}
        max_workers = min(8, len(expirations))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._fetch_options_for_expiration, symbol, exp): exp
                for exp in expirations
            }
            
            for future in concurrent.futures.as_completed(future_map):
                exp = future_map[future]
                try:
                    expiration, bucket = future.result()
                except Exception as e:
                    self.log(f"Error loading data for {exp}: {e}")
                    continue
                
                if bucket['calls'] or bucket['puts']:
                    grouped_options[expiration] = bucket
        
        self.log(f"Collected option data for {len(grouped_options)} expirations: ")
        self.log(json.dumps(grouped_options))
        return grouped_options
    
    def _get_expiration_close_price(self, symbol: str, expiration: str) -> Optional[float]:
        """Fetch the official close price for the expiration date (fallback to prior session)."""
        try:
            target_date = datetime.fromisoformat(expiration).date()
        except ValueError:
            self.log(f"Invalid expiration format for P&L: {expiration}")
            return None
        
        for _ in range(3):
            date_str = target_date.strftime('%Y-%m-%d')
            try:
                agg = self.client.get_daily_open_close_agg(ticker=symbol, date=date_str)
                close_price = getattr(agg, 'close', None)
                if close_price is None:
                    close_price = getattr(agg, 'after_hours', None)
                if close_price is not None:
                    return float(close_price)
            except Exception as e:
                self.log(f"Error fetching close for {symbol} on {date_str}: {e}")
            
            # Move to previous session if market was closed
            target_date -= timedelta(days=1)
        
        return None
    
    
    def calculate_black_scholes_probability(
        self,
        spot: float,
        strike: float,
        days_to_exp: int,
        volatility: Optional[float] = None
    ) -> float:
        """
        Probability that the underlying finishes below strike at expiration.
        Returns P(S_T < K) using the Black-Scholes risk-neutral distribution.
        """
        if days_to_exp <= 0:
            return 1.0 if spot <= strike else 0.0
        
        spot = max(spot, 0.01)
        strike = max(strike, 0.01)
        time_to_exp = max(days_to_exp / 365.0, 1e-4)
        
        if volatility is None or volatility <= 0:
            volatility = 0.30
        volatility = min(max(volatility, 0.05), 3.0)
        
        r = 0.045  # Approx current risk-free rate
        
        try:
            d2 = (math.log(spot / strike) + (r - 0.5 * volatility**2) * time_to_exp) / (
                volatility * math.sqrt(time_to_exp)
            )
        except (ValueError, ZeroDivisionError):
            return 0.0
        
        cdf = 0.5 * (1 + math.erf(-d2 / math.sqrt(2)))
        return max(0.0, min(1.0, cdf))
    
    def construct_iron_condors(
        self,
        symbol: str,
        spot_price: float,
        options_chain: Dict,
        expiration: str,
        min_vol: int = 5,
        min_oi: int = 25,
        greek_filters: Optional[Dict[str, Optional[Tuple[Optional[float], Optional[float]]]]] = None,
        iv_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> List[IronCondor]:
        """Construct all possible iron condors from options chain"""
        iron_condors = []
        calls = options_chain['calls']
        puts = options_chain['puts']
        
        if not calls or not puts:
            return iron_condors
        
        # Filter options with sufficient liquidity
        liquid_calls = sorted(
            [c for c in calls if c['volume'] >= min_vol and c['open_interest'] >= min_oi],
            key=lambda x: x['strike']
        )
        liquid_puts = sorted(
            [p for p in puts if p['volume'] >= min_vol and p['open_interest'] >= min_oi],
            key=lambda x: x['strike']
        )
        
        if self.debug:
            self.log(
                f"Expiration {expiration}: {len(calls)} calls -> {len(liquid_calls)} liquid, "
                f"{len(puts)} puts -> {len(liquid_puts)} liquid"
            )

        if len(liquid_calls) < 2 or len(liquid_puts) < 2:
            return iron_condors
        
        strike_window_upper = spot_price * (1 + STRIKE_DISTANCE_PCT)
        strike_window_lower = spot_price * (1 - STRIKE_DISTANCE_PCT)
        
        call_candidates = [
            c for c in liquid_calls
            if spot_price <= c['strike'] <= strike_window_upper
        ][:MAX_OPTIONS_PER_SIDE]
        
        put_candidates = [
            p for p in liquid_puts
            if strike_window_lower <= p['strike'] <= spot_price
        ]
        # Keep puts closest to spot
        put_candidates = put_candidates[-MAX_OPTIONS_PER_SIDE:]
        
        if len(call_candidates) < 2 or len(put_candidates) < 2:
            return iron_condors
        
        # Calculate days to expiration
        exp_date = datetime.fromisoformat(expiration).date()
        today = datetime.now(ET).date()
        days_to_exp = (exp_date - today).days
        
        if days_to_exp <= 0:
            return iron_condors
        
        max_spread_width = max(
            MIN_SPREAD_ABS,
            min(MAX_SPREAD_ABS, spot_price * MAX_SPREAD_PCT)
        )
        
        self.log(f"Constructing iron condors from {len(call_candidates)} calls and {len(put_candidates)} puts")
        
        put_candidates_desc = list(reversed(put_candidates))
        
        greek_filters = greek_filters or {}
        short_delta_range = greek_filters.get("short_delta")
        long_delta_range = greek_filters.get("long_delta")
        short_theta_range = greek_filters.get("short_theta")
        long_theta_range = greek_filters.get("long_theta")

        def leg_passes(option: Dict, delta_bounds, theta_bounds) -> bool:
            return (
                value_in_range(option.get('delta'), delta_bounds)
                and value_in_range(option.get('theta'), theta_bounds)
                and value_in_range(option.get('implied_volatility'), iv_range)
            )

        # Find iron condor combinations (optimized)
        combinations_checked = 0
        for i, call_sell in enumerate(call_candidates):
            for j, call_buy in enumerate(call_candidates[i+1:], i+1):  # Only higher strikes
                call_spread_width = call_buy['strike'] - call_sell['strike']
                if call_spread_width <= 0 or call_spread_width > max_spread_width:
                    continue
                if not leg_passes(call_sell, short_delta_range, short_theta_range):
                    continue
                if not leg_passes(call_buy, long_delta_range, long_theta_range):
                    continue
                
                for k, put_sell in enumerate(put_candidates_desc):
                    for l, put_buy in enumerate(put_candidates_desc[k+1:], k+1):  # Lower strikes
                        put_spread_width = put_sell['strike'] - put_buy['strike']
                        if put_spread_width <= 0 or put_spread_width > max_spread_width:
                            continue
                        if not leg_passes(put_sell, short_delta_range, short_theta_range):
                            continue
                        if not leg_passes(put_buy, long_delta_range, long_theta_range):
                            continue
                        
                        combinations_checked += 1
                        if combinations_checked % 1000 == 0:
                            self.log(f"Checked {combinations_checked} combinations...")
                        
                        # Ensure profit zone is reasonable
                        if put_sell['strike'] >= call_sell['strike']:
                            continue
                        
                        # Calculate net credit
                        call_credit = (call_sell['bid'] + call_sell['ask']) / 2
                        call_debit = (call_buy['bid'] + call_buy['ask']) / 2
                        put_credit = (put_sell['bid'] + put_sell['ask']) / 2
                        put_debit = (put_buy['bid'] + put_buy['ask']) / 2
                        
                        net_credit = call_credit - call_debit + put_credit - put_debit
                        
                        if net_credit <= 0:
                            continue
                        
                        # Calculate risk metrics
                        call_spread_width = call_buy['strike'] - call_sell['strike']
                        put_spread_width = put_sell['strike'] - put_buy['strike']
                        widest_spread = max(call_spread_width, put_spread_width)
                        
                        max_profit = net_credit
                        max_loss = widest_spread - net_credit
                        
                        if max_loss <= 0:
                            continue
                        credit_ratio = round(net_credit / widest_spread, 4) if widest_spread > 0 else 0.0
                        
                        vol_candidates = [
                            call_sell.get('implied_volatility'),
                            call_buy.get('implied_volatility'),
                            put_sell.get('implied_volatility'),
                            put_buy.get('implied_volatility'),
                        ]
                        vol_candidates = [v for v in vol_candidates if v and v > 0]
                        effective_vol = float(np.mean(vol_candidates)) if vol_candidates else None
                        
                        # Calculate probability of profit (stock stays in profit zone)
                        # PoP = Prob(Spot < Call Strike) - Prob(Spot < Put Strike)
                        profit_zone_lower = put_sell['strike']
                        profit_zone_upper = call_sell['strike']
                        
                        prob_below_call = self.calculate_black_scholes_probability(
                            spot_price,
                            profit_zone_upper,
                            days_to_exp,
                            effective_vol
                        )
                        prob_below_put = self.calculate_black_scholes_probability(
                            spot_price,
                            profit_zone_lower,
                            days_to_exp,
                            effective_vol
                        )
                        
                        prob_in_zone = max(0.0, min(1.0, prob_below_call - prob_below_put))
                        
                        risk_reward = max_profit / max_loss
                        
                        iron_condor = IronCondor(
                            expiration=expiration,
                            call_spread=(call_sell['strike'], call_buy['strike']),
                            put_spread=(put_sell['strike'], put_buy['strike']),
                            net_credit=round(net_credit, 2),
                            max_profit=round(max_profit, 2),
                            max_loss=round(max_loss, 2),
                            profit_zone=(profit_zone_lower, profit_zone_upper),
                            probability_of_profit=round(prob_in_zone * 100, 1),
                            risk_reward_ratio=round(risk_reward, 2),
                            days_to_expiration=days_to_exp,
                            spot_price=spot_price,
                            credit_ratio=credit_ratio,
                            call_sell_delta=call_sell.get('delta'),
                            call_sell_theta=call_sell.get('theta'),
                            call_sell_iv=call_sell.get('implied_volatility'),
                            call_sell_volume=call_sell.get('volume'),
                            call_sell_open_interest=call_sell.get('open_interest'),
                            call_buy_delta=call_buy.get('delta'),
                            call_buy_theta=call_buy.get('theta'),
                            call_buy_iv=call_buy.get('implied_volatility'),
                            call_buy_volume=call_buy.get('volume'),
                            call_buy_open_interest=call_buy.get('open_interest'),
                            put_sell_delta=put_sell.get('delta'),
                            put_sell_theta=put_sell.get('theta'),
                            put_sell_iv=put_sell.get('implied_volatility'),
                            put_sell_volume=put_sell.get('volume'),
                            put_sell_open_interest=put_sell.get('open_interest'),
                            put_buy_delta=put_buy.get('delta'),
                            put_buy_theta=put_buy.get('theta'),
                            put_buy_iv=put_buy.get('implied_volatility'),
                            put_buy_volume=put_buy.get('volume'),
                            put_buy_open_interest=put_buy.get('open_interest')
                        )
                        
                        iron_condors.append(iron_condor)
                        
                        # Limit total iron condors to prevent memory issues
                        if len(iron_condors) >= 1000:
                            self.log(f"Reached limit of 1000 iron condors, stopping construction")
                            return iron_condors
        
        self.log(f"Constructed {len(iron_condors)} iron condors from {combinations_checked} combinations")
        return iron_condors
    
    def find_best_iron_condors(
        self,
        symbol: str,
        max_days: int = 7,
        min_days: int = 5,
        min_net_credit: float = 0.10,
        max_risk: float = 10.00,
        min_probability: float = 30.0,
        limit: int = 10,
        min_vol: int = 5,
        min_oi: int = 25,
        greek_filters: Optional[Dict[str, Optional[Tuple[Optional[float], Optional[float]]]]] = None,
        iv_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        min_credit_ratio: float = 0.0,
        capital_limit: Optional[float] = None,
    ) -> Tuple[List[IronCondor], List[IronCondor], bool]:
        """Find the best iron condor opportunities"""
        print(f"🔍 Scanning {symbol} for iron condor opportunities...")
        
        max_days = max(max_days, min_days)

        # Get current price
        # spot_price = self.get_current_price(symbol) #todo: free tier limit - reverting to last close
        spot_price = self.get_last_close(symbol)
        if spot_price == 0:
            print(f"❌ Could not get current price for {symbol}")
            return [], False
        
        # Check for upcoming earnings //
        has_earnings = self.check_upcoming_earnings(symbol, max_days)
        if has_earnings: # todo: free tier limit, no earnings 
            print(f"⚠️  Warning: {symbol} has upcoming earnings within {max_days} days")
        
        print(f"💰 Current spot price: ${spot_price:.2f}")
        
        # Get all options grouped by expiration
        grouped_options = self.fetch_and_group_options(symbol, max_days)
        
        # Filter expirations by date
        today = datetime.now(ET).date()
        end_date = today + timedelta(days=max_days)
        min_date = today + timedelta(days=max(min_days, 0))
        
        valid_expirations = []
        for exp_str in grouped_options.keys():
            try:
                exp_date = datetime.fromisoformat(exp_str).date()
                if min_date <= exp_date <= end_date:
                    valid_expirations.append(exp_str)
            except:
                continue
                
        valid_expirations.sort()
        
        if not valid_expirations:
            print(f"❌ No expirations found for {symbol} within {max_days} days")
            return [], has_earnings
            
        print(f"📅 Found {len(valid_expirations)} available expirations")
        
        all_iron_condors = []
        
        # Helper function for parallel processing
        def process_expiration(expiration):
            self.log(f"Processing expiration: {expiration}")
            options_chain = grouped_options[expiration]
            
            if not options_chain['calls'] or not options_chain['puts']:
                self.log(f"No options found for {expiration}")
                return []
            
            return self.construct_iron_condors(
                symbol,
                spot_price,
                options_chain,
                expiration,
                min_vol,
                min_oi,
                greek_filters=greek_filters,
                iv_range=iv_range,
            )

        # Use ThreadPoolExecutor for concurrent processing (CPU bound now, but still good for separation)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_exp = {executor.submit(process_expiration, exp): exp for exp in valid_expirations}
            for future in concurrent.futures.as_completed(future_to_exp):
                try:
                    condors = future.result()
                    all_iron_condors.extend(condors)
                except Exception as e:
                    self.log(f"Error processing expiration: {e}")
        
        # Filter iron condors
        filtered_condors = []
        contract_multiplier = 100
        capital_limit = capital_limit if capital_limit and capital_limit > 0 else None
        min_credit_ratio = max(min_credit_ratio, 0.0)

        for ic in all_iron_condors:
            spread_width = ic.max_loss + ic.net_credit
            credit_ratio = ic.credit_ratio if ic.credit_ratio else (ic.net_credit / spread_width if spread_width else 0.0)
            per_contract_risk = ic.max_loss * contract_multiplier

            if ic.net_credit < min_net_credit:
                continue
            if ic.max_loss > max_risk:
                continue
            if ic.probability_of_profit < min_probability:
                continue
            if min_credit_ratio and credit_ratio < min_credit_ratio:
                continue
            if capital_limit and per_contract_risk > capital_limit:
                continue

            filtered_condors.append(ic)
        
        filter_summary = {
            'min_net_credit': min_net_credit,
            'max_risk': max_risk,
            'min_probability': f"{min_probability}%",
        }
        if min_credit_ratio:
            filter_summary['min_credit_ratio'] = min_credit_ratio
        if capital_limit:
            filter_summary['max_capital_per_condor'] = capital_limit
        print(f"🎯 Using filters: {filter_summary}")
        print(f"🏆 Found {len(filtered_condors)} total iron condors")
        
        csv_condors: List[IronCondor] = []
        display_condors: List[IronCondor] = []

        # Rank and limit results
        if filtered_condors:
            # Sort by net credit (highest first)
            filtered_condors.sort(key=lambda x: x.net_credit, reverse=True)
            csv_condors = list(filtered_condors)
            # Limit to top N for terminal display
            display_condors = filtered_condors[:limit]
            print(f"📊 Showing top {len(display_condors)} iron condors (ranked by net credit)")
        else:
            print("📊 Showing top 0 iron condors (ranked by net credit)")
        
        return display_condors, csv_condors, has_earnings
    
    def display_results(self, iron_condors: List[IronCondor], criteria: str = "credit"):
        """Display iron condor results"""
        if not iron_condors:
            print("❌ No iron condors found matching criteria")
            return
        
        # Sort by criteria
        if criteria == "credit":
            sorted_condors = sorted(iron_condors, key=lambda x: x.net_credit, reverse=True)
        elif criteria == "probability":
            sorted_condors = sorted(iron_condors, key=lambda x: x.probability_of_profit, reverse=True)
        elif criteria == "risk_reward":
            sorted_condors = sorted(iron_condors, key=lambda x: x.risk_reward_ratio, reverse=True)
        else:
            sorted_condors = iron_condors
        
        print(f"\n📊 Top {len(iron_condors)} by {criteria.title()} (highest first):")
        print("   " + "=" * 120)
        print("   Exp        Call Spread    Put Spread    Net Credit  Max Profit  Max Loss   PoP%   Risk/Reward  Credit%")
        print("   " + "-" * 120)
        
        for ic in sorted_condors[:10]:
            call_spread = f"${ic.call_spread[0]:.0f}/${ic.call_spread[1]:.0f}"
            put_spread = f"${ic.put_spread[0]:.0f}/${ic.put_spread[1]:.0f}"
            exp_short = ic.expiration.split('-')[1] + '-' + ic.expiration.split('-')[2]
            
            print(f"   {exp_short:<10} {call_spread:<12} {put_spread:<12} ${ic.net_credit:<9.2f} "
                  f"${ic.max_profit:<9.2f} ${ic.max_loss:<8.2f} {ic.probability_of_profit:<6.1f}% {ic.risk_reward_ratio:<10.2f} {ic.credit_ratio*100:<8.1f}%")
    
    def save_to_csv(self, iron_condors: List[IronCondor], symbol: str, has_earnings: bool = False) -> str:
        """Save iron condors to CSV file"""
        if not iron_condors:
            return ""
        
        data = []
        for ic in iron_condors:
            data.append({
                'symbol': symbol,
                'expiration': ic.expiration,
                'call_sell_strike': ic.call_spread[0],
                'call_buy_strike': ic.call_spread[1],
                'put_sell_strike': ic.put_spread[0],
                'put_buy_strike': ic.put_spread[1],
                'net_credit': ic.net_credit,
                'max_profit': ic.max_profit,
                'max_loss': ic.max_loss,
                'profit_zone_lower': ic.profit_zone[0],
                'profit_zone_upper': ic.profit_zone[1],
                'probability_of_profit': ic.probability_of_profit,
                'risk_reward_ratio': ic.risk_reward_ratio,
                'days_to_expiration': ic.days_to_expiration,
                'spot_price': ic.spot_price,
                'credit_ratio': ic.credit_ratio,
                'call_sell_delta': ic.call_sell_delta,
                'call_sell_theta': ic.call_sell_theta,
                'call_sell_iv': ic.call_sell_iv,
                'call_sell_volume': ic.call_sell_volume,
                'call_sell_open_interest': ic.call_sell_open_interest,
                'call_buy_delta': ic.call_buy_delta,
                'call_buy_theta': ic.call_buy_theta,
                'call_buy_iv': ic.call_buy_iv,
                'call_buy_volume': ic.call_buy_volume,
                'call_buy_open_interest': ic.call_buy_open_interest,
                'put_sell_delta': ic.put_sell_delta,
                'put_sell_theta': ic.put_sell_theta,
                'put_sell_iv': ic.put_sell_iv,
                'put_sell_volume': ic.put_sell_volume,
                'put_sell_open_interest': ic.put_sell_open_interest,
                'put_buy_delta': ic.put_buy_delta,
                'put_buy_theta': ic.put_buy_theta,
                'put_buy_iv': ic.put_buy_iv,
                'put_buy_volume': ic.put_buy_volume,
                'put_buy_open_interest': ic.put_buy_open_interest,
                'has_upcoming_earnings': has_earnings,
                'timestamp': datetime.now(ET).isoformat()
            })
        
        df = pd.DataFrame(data)
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        filename = f"data/{symbol.lower()}_iron_condors.csv"
        df.to_csv(filename, index=False)
        
        print(f"💾 Saved CSV file: {filename}")
        return filename
    
    def calculate_pnl(self, csv_path: str, closing_price: Optional[float] = None):
        """Calculate P&L for expired iron condors"""
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        if df.empty:
            print("❌ No data found in CSV file")
            return
        
        expiration_prices: Dict[Tuple[str, str], Optional[float]] = {}
        missing_prices: set = set()
        
        if closing_price is not None:
            print(f"📊 Calculating P&L for {len(df)} iron condors using provided closing price ${closing_price:.2f}...")
        else:
            print(f"📡 Fetching official closing prices for {len(df)} iron condors...")
        
        unique_keys = df[['symbol', 'expiration']].drop_duplicates()
        for _, key_row in unique_keys.iterrows():
            key = (key_row['symbol'], key_row['expiration'])
            if closing_price is not None:
                expiration_prices[key] = closing_price
                continue
            
            price = self._get_expiration_close_price(key_row['symbol'], key_row['expiration'])
            if price is not None:
                expiration_prices[key] = price
            else:
                expiration_prices[key] = None
                missing_prices.add(key_row['expiration'])
        
        if missing_prices and closing_price is None:
            print(f"⚠️  Unable to fetch closing prices for: {sorted(missing_prices)}. Skipping those rows.")
        
        total_trades = len(df)
        profitable_trades = 0
        total_pnl = 0.0
        evaluated_trades = 0
        pnl_details = []  # Store details for debugging
        
        for _, row in df.iterrows():
            key = (row['symbol'], row['expiration'])
            row_close = expiration_prices.get(key, closing_price)
            if row_close is None:
                continue
            
            # Extract strike prices
            call_sell = row['call_sell_strike']
            call_buy = row['call_buy_strike']
            put_sell = row['put_sell_strike']
            put_buy = row['put_buy_strike']
            net_credit = row['net_credit']
            
            # Calculate P&L based on closing price
            # 1. Profit Zone: Between sold strikes
            if put_sell <= row_close <= call_sell:
                pnl = net_credit
                profitable_trades += 1
                
            # 2. Call Side Loss
            elif row_close > call_sell:
                # Calculate loss on call spread
                # If price > call_buy: lose full spread width
                # If price between call_sell and call_buy: lose intrinsic value
                call_spread_width = call_buy - call_sell
                if row_close >= call_buy:
                    # Maximum loss on call side
                    loss_amt = call_spread_width
                else:
                    # Partial loss: intrinsic value of call spread
                    loss_amt = row_close - call_sell
                    
                pnl = net_credit - loss_amt
                
            # 3. Put Side Loss
            elif row_close < put_sell:
                # Calculate loss on put spread
                # If price < put_buy: lose full spread width
                # If price between put_buy and put_sell: lose intrinsic value
                put_spread_width = put_sell - put_buy
                if row_close <= put_buy:
                    # Maximum loss on put side
                    loss_amt = put_spread_width
                else:
                    # Partial loss: intrinsic value of put spread
                    loss_amt = put_sell - row_close
                    
                pnl = net_credit - loss_amt
            
            evaluated_trades += 1
            
            # Check if profitable (pnl > 0) - only count if not already counted in profit zone
            is_profitable = pnl > 0
            in_profit_zone = put_sell <= row_close <= call_sell
            if is_profitable and not in_profit_zone:
                # Can still be profitable if loss is less than credit received (already counted if in zone)
                profitable_trades += 1
            
            # Store details for summary
            pnl_details.append({
                'profit_zone': f"${put_sell:.1f}-${call_sell:.1f}",
                'close_price': row_close,
                'in_zone': in_profit_zone,
                'pnl': pnl,
                'profitable': is_profitable
            })
            
            total_pnl += pnl
        
        if evaluated_trades == 0:
            print("❌ Unable to evaluate P&L because no closing prices were available.")
            return
        
        win_rate = (profitable_trades / evaluated_trades) * 100 if evaluated_trades > 0 else 0
        avg_pnl = total_pnl / evaluated_trades if evaluated_trades > 0 else 0
        skipped_trades = total_trades - evaluated_trades
        closing_context = f"${closing_price:.2f}" if closing_price is not None else "per-expiration closes"
        
        # Show the actual closing prices used for debugging
        if closing_price is None and expiration_prices:
            unique_closes = set(v for v in expiration_prices.values() if v is not None)
            if len(unique_closes) == 1:
                closing_context = f"${list(unique_closes)[0]:.2f} (auto-fetched)"
            elif len(unique_closes) > 1:
                closing_context = f"multiple prices: {sorted(unique_closes)}"
        
        # Analyze P&L details
        in_zone_profitable = sum(1 for d in pnl_details if d['in_zone'] and d['profitable'])
        out_zone_profitable = sum(1 for d in pnl_details if not d['in_zone'] and d['profitable'])
        unprofitable = sum(1 for d in pnl_details if not d['profitable'])
        
        print(f"\n📈 P&L Summary (Closing Prices: {closing_context}):")
        print(f"   Trades Evaluated: {evaluated_trades} / {total_trades}")
        if skipped_trades:
            print(f"   Skipped Trades (missing price): {skipped_trades}")
        print(f"   Profitable Trades: {profitable_trades} ({in_zone_profitable} in profit zone, {out_zone_profitable} outside but still profitable)")
        print(f"   Unprofitable Trades: {unprofitable}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total P&L: ${total_pnl:.2f}")
        print(f"   Average P&L per Trade: ${avg_pnl:.2f}")
        
        # Warning if all trades are profitable (unusual)
        if evaluated_trades > 0 and profitable_trades == evaluated_trades:
            print(f"\n⚠️  WARNING: All {evaluated_trades} trades are profitable. This is unusual.")
            print(f"   Expected win rate based on PoP: ~30-35%")
            print(f"   Please verify the closing price used: {closing_context}")
            if pnl_details:
                sample_zone = pnl_details[0]['profit_zone']
                sample_close = pnl_details[0]['close_price']
                print(f"   Example: Profit zone {sample_zone}, Close: ${sample_close:.2f}")
        
        return {
            "evaluated_trades": evaluated_trades,
            "total_trades": total_trades,
            "skipped_trades": skipped_trades,
            "profitable_trades": profitable_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_pnl": avg_pnl,
        }

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Iron Condor Screener & Analyzer")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Find command
    find_parser = subparsers.add_parser('find', help='Find iron condor opportunities')
    find_parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., SPY)')
    find_parser.add_argument('--max-days', type=int, default=7, help='Maximum days to expiration - default: 7')
    find_parser.add_argument('--min-days', type=int, default=5, help='Minimum days to expiration - default: 5')
    find_parser.add_argument('--min-credit', type=float, default=0.10, help='Minimum net credit - default: 0.10')
    find_parser.add_argument('--max-risk', type=float, default=10.00, help='Maximum risk - default: 10.00')
    find_parser.add_argument('--min-probability', type=float, default=30.0, help='Minimum probability of profit percent - default: 30.0')
    find_parser.add_argument('--min-vol', type=int, default=5, help='Minimum volume per option - default: 5')
    find_parser.add_argument('--min-oi', type=int, default=25, help='Minimum open interest per option - default: 25')
    find_parser.add_argument('--short-delta-range', type=parse_range, help='Delta range for short legs (e.g., -0.35,-0.15)')
    find_parser.add_argument('--long-delta-range', type=parse_range, help='Delta range for long legs (e.g., -0.1,0.1)')
    find_parser.add_argument('--short-theta-range', type=parse_range, help='Theta range for short legs')
    find_parser.add_argument('--long-theta-range', type=parse_range, help='Theta range for long legs')
    find_parser.add_argument('--iv-range', type=parse_range, help='Implied volatility range (applied to every leg)')
    find_parser.add_argument('--min-credit-ratio', type=float, default=0.0, help='Minimum credit / spread width (0 - 1)')
    find_parser.add_argument('--account-size', type=float, help='Account size in USD for capital %% calculations')
    find_parser.add_argument('--max-capital-pct', type=float, default=5.0, help='Max percent of account per condor (default: 5%% when account size is set)')
    find_parser.add_argument('--max-capital', type=float, help='Absolute USD cap per condor (per contract risk)')
    find_parser.add_argument('--criteria', choices=['credit', 'probability', 'risk_reward'], 
                           default='credit', help='Ranking criteria - default: credit')
    find_parser.add_argument('--limit', type=int, default=10, help='Maximum number of iron condors to save to CSV - default: 10')
    
    # P&L command
    pnl_parser = subparsers.add_parser('pnl', help='Calculate P&L for expired iron condors')
    pnl_parser.add_argument('--csv', required=True, help='Path to CSV file with iron condor data')
    pnl_parser.add_argument('--closing-price', type=float, help='Override closing price (auto-fetched when omitted)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        screener = IronCondorScreener()
        
        if args.command == 'find':
            greek_filters = {
                "short_delta": args.short_delta_range,
                "long_delta": args.long_delta_range,
                "short_theta": args.short_theta_range,
                "long_theta": args.long_theta_range,
            }

            iv_range = args.iv_range

            capital_limit = args.max_capital
            pct_limit = None
            if args.account_size and args.max_capital_pct:
                pct_limit = args.account_size * (args.max_capital_pct / 100.0)
            if capital_limit is None:
                capital_limit = pct_limit
            elif pct_limit is not None:
                capital_limit = min(capital_limit, pct_limit)

            display_condors, csv_condors, has_earnings = screener.find_best_iron_condors(
                symbol=args.symbol.upper(),
                max_days=args.max_days,
                min_days=args.min_days,
                min_net_credit=args.min_credit,
                max_risk=args.max_risk,
                min_probability=args.min_probability,
                min_credit_ratio=args.min_credit_ratio,
                limit=args.limit,
                min_vol=args.min_vol,
                min_oi=args.min_oi,
                greek_filters=greek_filters,
                iv_range=iv_range,
                capital_limit=capital_limit,
            )
            
            if display_condors:
                screener.display_results(display_condors, args.criteria)
                csv_path = screener.save_to_csv(csv_condors, args.symbol.upper(), has_earnings)
                
                if display_condors:
                    best = display_condors[0]
                    print(f"\n🎯 Top Recommendation: {best.expiration} "
                          f"${best.call_spread[0]:.0f}/${best.call_spread[1]:.0f} call spread + "
                          f"${best.put_spread[0]:.0f}/${best.put_spread[1]:.0f} put spread")
                    print(f"   Net Credit: ${best.net_credit:.2f} | Max Profit: ${best.max_profit:.2f} | "
                          f"PoP: {best.probability_of_profit:.1f}% | Risk/Reward: {best.risk_reward_ratio:.2f}")
                
                print(f"\n💡 Next step: Run 'uv run screener.py pnl --csv {csv_path}' after expiration")
        
        elif args.command == 'pnl':
            screener.calculate_pnl(args.csv, args.closing_price)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if os.getenv('DEBUG', 'false').lower() == 'true':
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()