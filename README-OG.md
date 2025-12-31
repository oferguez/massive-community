# Iron Condor:

* [Spec](https://massive.com/blog/build-an-iron-condor-screener-with-massive)

* Notebook: 

* Example:

1) Short PUT $90 
1) Long PUT $98 
1) Short CALL $102
1) Long CALL $110

* 4 Options, Mkt at time of purchase: $100

|Type|Side |Strike|Premium|
|----|-----|------|-------|
|PUT |Long |$90   |$1.00  |
|PUT |Short|$98   |$2.50  |
|CALL|Short|$102  |$1.50  |
|CALL|Long |$110  |$0.75  |

* Returns per Mkt price

|Market Now|$80.00|$90.00|$99.00|$100.00|$101.00|$110.00|$120.00|$130.00|$140.00|$150.00|
|----------|------|------|------|-------|-------|-------|-------|-------|-------|-------|
|          |$10.00|$0.00 |$0.00 |$0.00  |$0.00  |$0.00  |$0.00  |$0.00  |$0.00  |$0.00  |
|          |-$18.00|-$8.00|$0.00 |$0.00  |$0.00  |$0.00  |$0.00  |$0.00  |$0.00  |$0.00  |
|          |$0.00 |$0.00 |$0.00 |$0.00  |$0.00  |-$8.00 |-$18.00|-$28.00|-$38.00|-$48.00|
|          |$0.00 |$0.00 |$0.00 |$0.00  |$0.00  |$0.00  |$10.00 |$20.00 |$30.00 |$40.00 |
|Position Cash Flow|-$8.00|-$8.00|$0.00 |$0.00  |$0.00  |-$8.00 |-$8.00 |-$8.00 |-$8.00 |-$8.00 |
|Total Profit|-$5.75|-$5.75|$2.25 |$2.25  |$2.25  |-$5.75 |-$5.75 |-$5.75 |-$5.75 |-$5.75 |



# Conda environment

* Created: 

  ```conda create -n massive-default python=3.11 numpy pandas matplotlib jupyter -y```

* Activated: To activate this environment: 

  ```$ conda activate massive-default```

* To deactivate an active environment: 

  ```$ conda deactivate```

* Snapshot:

  ```conda env export -n massive-default > ~/wsl_repos/massive-community/environment.yml```

* default Kernel set in settings.json:
```json
    {
      "python.defaultInterpreterPath": "/home/ofer/miniconda3/envs/massive-default/bin/python"
    }
```