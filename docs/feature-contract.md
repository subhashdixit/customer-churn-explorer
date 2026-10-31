# Feature contract

All named columns are required in CSVs. A blank value means missing; extra columns are ignored. Defaults are synthetic training medians/modes. Numeric bounds constrain serving and mark unsupported source values missing during training. They do not define a population validity guarantee.

| Source field | Meaning / units | Accepted values |
|---|---|---|
| `CreditScore` | CreditScore | 300 to 900 |
| `Geography` | Geography | France, Germany, Spain |
| `Age` | Age | 18 to 100 |
| `Tenure` | Tenure | 0 to 60 |
| `Balance` | Balance | 0 to 100000000.0 |
| `NumOfProducts` | NumOfProducts | 1 to 10 (whole numbers) |
| `HasCrCard` | HasCrCard | 0, 1 |
| `IsActiveMember` | IsActiveMember | 0, 1 |
| `EstimatedSalary` | EstimatedSalary | 0 to 100000000.0 |
