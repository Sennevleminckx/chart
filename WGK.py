#!/usr/bin/env python3
"""
preprocess.py: Parse input CSVs, compute summary stats, and save outputs.
Usage:
    python preprocess.py
Or with explicit files:
    python preprocess.py \
        --mapping mapping_file.csv \
        --items "Item-Tabel 1.csv" \
        --subdomains "Subdomain-Tabel 1.csv" \
        --responses transposed_data.csv \
        --outdir data/
Note: Searches files relative to script location.
"""
import os
import argparse
import pandas as pd

# ----------------------------------------------
# Interquartile range helper
# ----------------------------------------------
def iqr(x):
    "Interquartile range: Q3 - Q1"
    return x.quantile(0.75) - x.quantile(0.25)

# ----------------------------------------------
# Main processing function
# ----------------------------------------------
def main():
    # Default filenames in script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    defaults = {
        'mapping': 'mapping_file.csv',
        'items': 'Item-Tabel 1.csv',
        'subdomains': 'Subdomain-Tabel 1.csv',
        'responses': 'transposed_data.csv',
        'outdir': 'data/'
    }
    parser = argparse.ArgumentParser(
        description="Compute domain & subdomain stats from survey data"
    )
    parser.add_argument("--mapping",    default=defaults['mapping'],    help="CSV with question_code mapping (df1)")
    parser.add_argument("--items",      default=defaults['items'],      help="Item-Tabel CSV with code->subdomain_id (df2)")
    parser.add_argument("--subdomains", default=defaults['subdomains'], help="Subdomain-Tabel CSV with id->domainId (df3)")
    parser.add_argument("--responses",  default=defaults['responses'],  help="Transposed survey responses CSV (df4)")
    parser.add_argument("--outdir",     default=defaults['outdir'],     help="Directory to store output parquet files")
    args = parser.parse_args()

    # Change cwd to script directory
    os.chdir(script_dir)
    os.makedirs(args.outdir, exist_ok=True)

    # Build file paths
    mapping_path    = os.path.join(script_dir, args.mapping)
    items_path      = os.path.join(script_dir, args.items)
    subdomains_path = os.path.join(script_dir, args.subdomains)
    responses_path  = os.path.join(script_dir, args.responses)

    # Verify input files exist
    for p in (mapping_path, items_path, subdomains_path, responses_path):
        if not os.path.exists(p):
            parser.error(f"Input file not found: {p}")

    # 1) Load data
    df1 = pd.read_csv(mapping_path,    delimiter=',')
    df2 = pd.read_csv(items_path,      delimiter=';')
    df3 = pd.read_csv(subdomains_path, delimiter=';')
    df4 = pd.read_csv(responses_path,  delimiter=',')

    # 2) Merge mapping to get subdomain_id
    df1 = (
        df1
        .merge(df2[['code','subdomain_id']], how='left',
               left_on='question_code', right_on='code')
        .drop(columns='code')
    )

    # 3) Manual mapping fallback
    manual_map = {
        'ZT117':7, 'ZT142':6, 'ZT121':14, 'ZT123':14, 'ZT144':12,
        'ZT118':24,'ZT125':18,'ZT126':18,'ZT127':19,'ZT134':1,
        'ZT135':1, 'ZT136':1, 'ZT143':13,'ZT119':24,'ZT129':15,
        'ZT124':18,'ZT128':16,'ZT122':14,'ZT131':11,'ZT145':24,
        'ZT132':19,'ZT74':19, 'ZT133':1, 'ZT130':15,'ZT146':20,
        'ZT120':13,'ZT137':3, 'ZT138':3, 'ZT139':3, 'ZT140':3,
        'ZT141':3
    }
    df1['subdomain_id'] = (
        df1['subdomain_id']
        .fillna(df1['question_code'].map(manual_map))
    )

    # 4) Merge to get domainId(s)
    df1 = (
        df1
        .merge(df3[['id','domainId']], how='left',
               left_on='subdomain_id', right_on='id')
        .drop(columns='id')
    )

    # 5) Explode multi-domain IDs
    df1['domainId'] = df1['domainId'].astype(str).str.split(',')
    df1 = df1.explode('domainId').reset_index(drop=True)
    df1['domainId'] = df1['domainId'].astype(int)

    # 6) Melt responses to long format and annotate
    df_long = (
        df4
        .melt(id_vars=['employee_id','team'],
              var_name='question_code', value_name='score')
        .merge(df1[['question_code','subdomain_id','domainId']],
               on='question_code', how='left')
    )

    # 6a) Recode reversed items (1↔10) for specific questions
    reverse_items = [
        'ZT101','ZT102','ZT103','ZT13','ZT14','ZT15','ZT16','ZT18','ZT19',
        'ZT20','ZT21','ZT22','ZT23','ZT24','ZT25','ZT26','ZT28','ZT29',
        'ZT30','ZT33','ZT34','ZT4','ZT43','ZT60','ZT70','ZT74','ZT78',
        'ZT79','ZT80','ZT81','ZT85','ZT89'
    ]
    mask = df_long['question_code'].isin(reverse_items)
    # Assuming scores are integers 1–10
    df_long.loc[mask, 'score'] = 11 - df_long.loc[mask, 'score']

    # 7) Compute subdomain-level stats
    stats_df = (
        df_long
        .groupby(['team','domainId','subdomain_id'])['score']
        .agg(median_score='median', mean_score='mean',
             std_score='std', IQR_score=iqr)
        .reset_index()
    )

    # 8) Pivot wide for convenience
    pivot_df = stats_df.pivot_table(
        index=['domainId','subdomain_id'],
        columns='team',
        values=['median_score','mean_score','std_score','IQR_score']
    ).sort_index()

    # 9) Compute domain-level aggregation
    domain_stats = (
        df_long
        .groupby(['team','domainId'])['score']
        .agg(median_score=lambda x: x.quantile(0.5),
             mean_score='mean', std_score='std', IQR_score=iqr)
        .reset_index()
    )

    # 10) Save outputs to outdir
    # Save long-format data for front-end use
    df_long.to_parquet(
        os.path.join(args.outdir, 'df_long.parquet'),
        compression='snappy'
    )
    # Save precomputed stats
    domain_stats.to_parquet(
        os.path.join(args.outdir,'domain_stats.parquet'),
        compression='snappy'
    )
    stats_df.to_parquet(
        os.path.join(args.outdir,'sub_stats.parquet'),
        compression='snappy'
    )
    pivot_df.to_parquet(
        os.path.join(args.outdir,'pivot_df.parquet'),
        compression='snappy'
    )

    print(f"Saved df_long, domain_stats, sub_stats, pivot_df to '{args.outdir}'")

if __name__ == '__main__':
    main()
