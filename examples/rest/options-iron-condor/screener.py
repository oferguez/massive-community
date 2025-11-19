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
import concurrent.futures
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from massive import RESTClient
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Configuration
ET = ZoneInfo("America/New_York")
load_dotenv()

MAX_OPTIONS_PER_SIDE = 30
STRIKE_DISTANCE_PCT = 0.2
MAX_SPREAD_PCT = 0.08
MAX_SPREAD_ABS = 50.0
MIN_SPREAD_ABS = 0.5

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
                
                option_data = {
                    'strike': float(option.details.strike_price) if option.details.strike_price else 0.0,
                    'bid': float(option.last_quote.bid) if option.last_quote and option.last_quote.bid else 0.0,
                    'ask': float(option.last_quote.ask) if option.last_quote and option.last_quote.ask else 0.0,
                    'volume': int(option.day.volume) if option.day and option.day.volume else 0,
                    'open_interest': int(option.open_interest) if option.open_interest else 0,
                    'implied_volatility': self._extract_implied_volatility(option),
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
        
        self.log(f"Collected option data for {len(grouped_options)} expirations")
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
    
    def construct_iron_condors(self, symbol: str, spot_price: float, 
                             options_chain: Dict, expiration: str,
                             min_vol: int = 5, min_oi: int = 25) -> List[IronCondor]:
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
        
        # Find iron condor combinations (optimized)
        combinations_checked = 0
        for i, call_sell in enumerate(call_candidates):
            for j, call_buy in enumerate(call_candidates[i+1:], i+1):  # Only higher strikes
                call_spread_width = call_buy['strike'] - call_sell['strike']
                if call_spread_width <= 0 or call_spread_width > max_spread_width:
                    continue
                
                for k, put_sell in enumerate(put_candidates_desc):
                    for l, put_buy in enumerate(put_candidates_desc[k+1:], k+1):  # Lower strikes
                        put_spread_width = put_sell['strike'] - put_buy['strike']
                        if put_spread_width <= 0 or put_spread_width > max_spread_width:
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
                            spot_price=spot_price
                        )
                        
                        iron_condors.append(iron_condor)
                        
                        # Limit total iron condors to prevent memory issues
                        if len(iron_condors) >= 1000:
                            self.log(f"Reached limit of 1000 iron condors, stopping construction")
                            return iron_condors
        
        self.log(f"Constructed {len(iron_condors)} iron condors from {combinations_checked} combinations")
        return iron_condors
    
    def find_best_iron_condors(self, symbol: str, max_days: int = 7, 
                             min_net_credit: float = 0.10, max_risk: float = 10.00,
                             min_probability: float = 30.0, limit: int = 10,
                             min_vol: int = 5, min_oi: int = 25) -> Tuple[List[IronCondor], bool]:
        """Find the best iron condor opportunities"""
        print(f"🔍 Scanning {symbol} for iron condor opportunities...")
        
        # Get current price
        spot_price = self.get_current_price(symbol)
        if spot_price == 0:
            print(f"❌ Could not get current price for {symbol}")
            return [], False
        
        # Check for upcoming earnings
        has_earnings = self.check_upcoming_earnings(symbol, max_days)
        if has_earnings:
            print(f"⚠️  Warning: {symbol} has upcoming earnings within {max_days} days")
        
        print(f"💰 Current spot price: ${spot_price:.2f}")
        
        # Get all options grouped by expiration
        grouped_options = self.fetch_and_group_options(symbol, max_days)
        
        # Filter expirations by date
        today = datetime.now(ET).date()
        end_date = today + timedelta(days=max_days)
        
        valid_expirations = []
        for exp_str in grouped_options.keys():
            try:
                exp_date = datetime.fromisoformat(exp_str).date()
                if today <= exp_date <= end_date:
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
            
            return self.construct_iron_condors(symbol, spot_price, options_chain, expiration, min_vol, min_oi)

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
        filtered_condors = [
            ic for ic in all_iron_condors
            if ic.net_credit >= min_net_credit 
            and ic.max_loss <= max_risk
            and ic.probability_of_profit >= min_probability
        ]
        
        print(f"🎯 Using filters: {{'min_net_credit': {min_net_credit}, 'max_risk': {max_risk}, 'min_probability': {min_probability}%}}")
        print(f"🏆 Found {len(filtered_condors)} total iron condors")
        
        # Rank and limit results
        if filtered_condors:
            # Sort by net credit (highest first)
            filtered_condors.sort(key=lambda x: x.net_credit, reverse=True)
            # Limit to top N results
            filtered_condors = filtered_condors[:limit]
            print(f"📊 Showing top {len(filtered_condors)} iron condors (ranked by net credit)")
        
        return filtered_condors, has_earnings
    
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
        print("   " + "=" * 100)
        print("   Exp        Call Spread    Put Spread    Net Credit  Max Profit  Max Loss   PoP%   Risk/Reward")
        print("   " + "-" * 100)
        
        for ic in sorted_condors[:5]:
            call_spread = f"${ic.call_spread[0]:.0f}/${ic.call_spread[1]:.0f}"
            put_spread = f"${ic.put_spread[0]:.0f}/${ic.put_spread[1]:.0f}"
            exp_short = ic.expiration.split('-')[1] + '-' + ic.expiration.split('-')[2]
            
            print(f"   {exp_short:<10} {call_spread:<12} {put_spread:<12} ${ic.net_credit:<9.2f} "
                  f"${ic.max_profit:<9.2f} ${ic.max_loss:<8.2f} {ic.probability_of_profit:<6.1f}% {ic.risk_reward_ratio:<10.2f}")
    
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
                # Loss = (Price - Sold Strike) - Credit
                # Capped at (Buy - Sold) - Credit
                loss_amt = (row_close - call_sell)
                max_loss_call = call_buy - call_sell
                
                if loss_amt > max_loss_call:
                    loss_amt = max_loss_call
                    
                pnl = net_credit - loss_amt
                
            # 3. Put Side Loss
            elif row_close < put_sell:
                # Loss = (Sold Strike - Price) - Credit
                # Capped at (Sold - Buy) - Credit
                loss_amt = (put_sell - row_close)
                max_loss_put = put_sell - put_buy
                
                if loss_amt > max_loss_put:
                    loss_amt = max_loss_put
                    
                pnl = net_credit - loss_amt
            
            evaluated_trades += 1
            
            # Check if profitable (pnl > 0)
            if pnl > 0 and not (put_sell <= row_close <= call_sell):
                # Can still be profitable if loss is less than credit received
                profitable_trades += 1
            
            total_pnl += pnl
        
        if evaluated_trades == 0:
            print("❌ Unable to evaluate P&L because no closing prices were available.")
            return
        
        win_rate = (profitable_trades / evaluated_trades) * 100
        avg_pnl = total_pnl / evaluated_trades
        skipped_trades = total_trades - evaluated_trades
        closing_context = f"${closing_price:.2f}" if closing_price is not None else "per-expiration closes"
        
        print(f"\n📈 P&L Summary (Closing Prices: {closing_context}):")
        print(f"   Trades Evaluated: {evaluated_trades} / {total_trades}")
        if skipped_trades:
            print(f"   Skipped Trades (missing price): {skipped_trades}")
        print(f"   Profitable Trades: {profitable_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total P&L: ${total_pnl:.2f}")
        print(f"   Average P&L per Trade: ${avg_pnl:.2f}")
        
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
    find_parser.add_argument('--min-credit', type=float, default=0.10, help='Minimum net credit - default: 0.10')
    find_parser.add_argument('--max-risk', type=float, default=10.00, help='Maximum risk - default: 10.00')
    find_parser.add_argument('--min-probability', type=float, default=30.0, help='Minimum probability of profit percent - default: 30.0')
    find_parser.add_argument('--min-vol', type=int, default=5, help='Minimum volume per option - default: 5')
    find_parser.add_argument('--min-oi', type=int, default=25, help='Minimum open interest per option - default: 25')
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
            iron_condors, has_earnings = screener.find_best_iron_condors(
                symbol=args.symbol.upper(),
                max_days=args.max_days,
                min_net_credit=args.min_credit,
                max_risk=args.max_risk,
                min_probability=args.min_probability,
                limit=args.limit,
                min_vol=args.min_vol,
                min_oi=args.min_oi
            )
            
            if iron_condors:
                screener.display_results(iron_condors, args.criteria)
                csv_path = screener.save_to_csv(iron_condors, args.symbol.upper(), has_earnings)
                
                if iron_condors:
                    best = iron_condors[0]
                    print(f"\n🎯 Top Recommendation: {best.expiration} "
                          f"${best.call_spread[0]:.0f}/${best.call_spread[1]:.0f} call spread + "
                          f"${best.put_spread[0]:.0f}/${best.put_spread[1]:.0f} put spread")
                    print(f"   Net Credit: ${best.net_credit:.2f} | Max Profit: ${best.max_profit:.2f} | "
                          f"PoP: {best.probability_of_profit:.1f}% | Risk/Reward: {best.risk_reward_ratio:.2f}")
                
                print(f"\n💡 Next step: Run 'python screener.py pnl --csv {csv_path}' after expiration")
        
        elif args.command == 'pnl':
            screener.calculate_pnl(args.csv, args.closing_price)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if os.getenv('DEBUG', 'false').lower() == 'true':
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
