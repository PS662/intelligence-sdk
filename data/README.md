# data/

Sample datasets live here and are **not** committed (`data/*.csv` is git-ignored).
Generate the iris sample used by the recipes in `configs/`:

```bash
python tools/make_iris_csv.py          # writes data/iris.csv (150 rows, 4 numeric features, species)
```

The test suite generates the file itself when it is missing.
