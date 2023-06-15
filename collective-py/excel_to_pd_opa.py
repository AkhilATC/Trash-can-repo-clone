import pandas as pd
import io

# file to buffer
s = io.StringIO()
with open('names.csv') as file:
    for line in file:
        s.write(line)
s.seek(0)


# debunk pandas read csv
def pandas_csv_operations():
    """
    pandas.read_csv(
        filepath_or_buffer,
        *,
        sep=_NoDefault.no_default,
        delimiter=None,
        header='infer',
        names=_NoDefault.no_default,
        index_col=None,
        usecols=None,
        squeeze=None,
        prefix=_NoDefault.no_default,
        mangle_dupe_cols=True,
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skipinitialspace=False,
        skiprows=None,
        skipfooter=0,
        nrows=None,
        na_values=None,
        keep_default_na=True,
        na_filter=True,
        verbose=False,
        skip_blank_lines=True,
        parse_dates=None,
        infer_datetime_format=False,
        keep_date_col=False,
        date_parser=None,
        dayfirst=False,
        cache_dates=True,
        iterator=False,
        chunksize=None,
        compression='infer',
        thousands=None,
        decimal='.',
        lineterminator=None,
        quotechar='"',
        quoting=0,
        doublequote=True,
        escapechar=None,
        comment=None,
        encoding=None,
        encoding_errors='strict',
        dialect=None,
        error_bad_lines=None,
        warn_bad_lines=None,
        on_bad_lines=None,
        delim_whitespace=False,
        low_memory=True,
        memory_map=False,
        float_precision=None,
        storage_options=None)

    """
    df = pd.read_excel('one.xlsx')
    mapped = pd.Series({col: list(df[col].unique()) for col in df})
    print(mapped.to_dict())
    # print(df.to_dict())


if __name__ == '__main__':
    pandas_csv_operations()