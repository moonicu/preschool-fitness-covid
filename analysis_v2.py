"""
Pre- vs post-pandemic differences in physical fitness among Korean preschool
children: serial cross-sectional analysis of Seoul daycare surveillance records
(2017-2019 vs 2022-2024).

Variables of the input extract
    covid  0 = 2017-2019, 1 = 2022-2024        time   programme cumulative measurement index
    year   calendar year of measurement         sex    1 = boy, 2 = girl
    age    age in completed months              height (cm), weight (kg)
    endu   V-sit endurance (s)                  flex   sit-and-reach (cm)
    equi   single-leg stand (s)                 quick  standing long jump (cm)
    agil   5-metre shuttle run (s; lower = better)

Processing (identical in both periods)
    plausibility limits -> zero values set to missing (sit-and-reach: only in
    records with all five tests 0; pre-pandemic zeros restored) -> 120-s ceiling for
    V-sit and single-leg stand -> values > 3 SD from the mean for age in
    months (periods pooled) set to missing.

Analyses
    OLS of each test on period, sex and a cubic B-spline of age (HC3 SE);
    standardised difference = adjusted difference / pre-period SD, oriented so
    that positive = post-pandemic deficit (percentile and quantile-regression
    differences are also given in this orientation, as 'deficit' columns). Age-group, sex x BMI-category and
    interaction analyses; percentiles within 6-month age bands with bootstrap
    CIs (quantile regression as a complementary analysis); sensitivity analyses; missingness
    audit; linkage to the programme's 2017-2019 repeat-measurement file
    (optional; aggregate counts only).

Usage
    PFN_DATA=/path/to/extract.csv [PFN_PRIOR=/path/to/prior.csv] [PFN_REPEAT=/path/to/file] python analysis_v2.py
    Paths may instead be placed in local_paths.json next to this script
    ({"PFN_DATA": "...", "PFN_PRIOR": "...", "PFN_REPEAT": "..."}) or the data put at data/data3.csv.
    Outputs are written to output_v2/{tables,supplementary,figures}.
"""
import os, sys, json, warnings
warnings.simplefilter('default')                       # show each distinct warning once
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices, build_design_matrices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

# ----------------------------------------------------------------------------
# Paths / config
# ----------------------------------------------------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
_LOCAL    = os.path.join(HERE, 'local_paths.json')   # optional, not distributed
_LOCAL_PATHS = {}
if os.path.exists(_LOCAL):
    with open(_LOCAL, encoding='utf-8') as fh:
        _LOCAL_PATHS = json.load(fh)

def _resolve(key, default):
    """Path from an environment variable, else local_paths.json, else the default."""
    return os.environ.get(key) or _LOCAL_PATHS.get(key) or default

# Raw extract (not redistributable)
DATA_PATH = _resolve('PFN_DATA', os.path.join(HERE, 'data', 'data3.csv'))
# Input file of the earlier report (optional: 2017-2019 sit-and-reach zeros)
PRIOR_PATH = _resolve('PFN_PRIOR', os.path.join(HERE, 'data', 'prior_extract.csv'))
# Programme's 2017-2019 repeat-measurement file, one row per child (optional; path given only via PFN_REPEAT)
REPEAT_PATH = _resolve('PFN_REPEAT', '')
REF_DIR   = os.path.join(HERE, 'ref')
OUT       = os.path.join(HERE, 'output_v2')
TAB, SUP, FIG = [os.path.join(OUT, d) for d in ('tables', 'supplementary', 'figures')]
for d in (OUT, TAB, SUP, FIG):
    os.makedirs(d, exist_ok=True)


# Corrected test dictionary --------------------------------------------------
TESTS = {   # plausible = physiological plausibility limits applied before all other processing
    'endu':  dict(name='V-Sit Endurance',      abbr='VSE', unit='s',  higher_better=True,  cap=120,  plausible=(0, 200)),
    'flex':  dict(name='Sit and Reach',        abbr='SRT', unit='cm', higher_better=True,  cap=None, plausible=(-30, 50)),
    'equi':  dict(name='Single-Leg Stand',     abbr='SLS', unit='s',  higher_better=True,  cap=120,  plausible=(0, 150)),
    'quick': dict(name='Standing Long Jump',   abbr='SLJ', unit='cm', higher_better=True,  cap=None, plausible=(0, 200)),
    'agil':  dict(name='5-metre Shuttle Run',  abbr='5SR', unit='s',  higher_better=False, cap=None, plausible=(0, 50)),
}
TEST_COLS = list(TESTS)
def label(c): return f"{TESTS[c]['name']} ({TESTS[c]['unit']})"

AGE_MIN, AGE_MAX = 36, 84                           # analysed age range (months)
AGE_GROUPS = [(lo, lo + 11) for lo in range(AGE_MIN, 72, 12)] + [(72, AGE_MAX)]   # 12-month groups; the last spans 13 months
def age_group(a):
    for lo, hi in AGE_GROUPS:
        if lo <= a <= hi:
            return f'{lo}–{hi}'
    return np.nan

# Cleaning rules (primary) ---------------------------------------------------
RULES_PRIMARY = dict(zero_to_na=True, cap=True, z_cut=3.0)
# Sit-and-reach: 0 cm is a valid score. Its zeros are kept unless all five tests of the record are 0
# (a record without measurements); zeros of the other tests are set to missing (rules['zero_to_na']).

# ----------------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------------
def load_raw():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    df.replace('null', np.nan, inplace=True)
    missing = [c for c in ['covid', 'time', 'year', 'sex', 'age', 'height', 'weight'] + TEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'input file lacks columns: {missing}')
    for c in ['covid', 'time', 'year', 'sex', 'age', 'height', 'weight'] + TEST_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # period coding must agree with the calendar year: 0 = 2017-2019, 1 = 2022-2024
    bad = ~(((df.covid == 0) & df.year.between(2017, 2019)) | ((df.covid == 1) & df.year.between(2022, 2024)))
    if bad.any():
        raise ValueError(f'{int(bad.sum())} records with period code inconsistent with year (or outside 2017-2019/2022-2024)')
    if not df.sex.dropna().isin([1, 2]).all():
        raise ValueError('sex must be coded 1 (boy) or 2 (girl)')
    return df


def clean(df_raw, rules=RULES_PRIMARY):
    """Return the analytic dataframe, a flow dictionary and a per-test processing audit."""
    flow = {}
    df = df_raw.copy()
    flow['raw_records'] = len(df)
    flow['raw_by_year'] = {int(y): int(n) for y, n in df['year'].value_counts().sort_index().items()}
    df = df[df['sex'].isin([1, 2]) & df['age'].notna()]
    flow['after_missing_sex_age'] = len(df)
    df = df[(df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX)].copy()
    flow['after_age_36_84'] = len(df)
    all_zero = (df[TEST_COLS] == 0).all(axis=1)          # records without measurements (all five tests 0)
    df['period']    = np.where(df['covid'] == 1, 'post', 'pre')
    df['post']      = (df['covid'] == 1).astype(int)
    df['female']    = (df['sex'] == 2).astype(int)
    df['age_group'] = df['age'].apply(age_group)
    df['year']      = df['year'].astype(int)

    # --- anthropometrics: implausible values then age-specific |z|>3 (pooled)
    df.loc[~df['height'].between(70, 150), 'height'] = np.nan
    df.loc[~df['weight'].between(8, 60),   'weight'] = np.nan
    for c in ['height', 'weight']:
        z = df.groupby('age')[c].transform(lambda x: (x - x.mean()) / x.std())
        df.loc[z.abs() > 3, c] = np.nan
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

    # --- fitness tests: identical rule in both periods
    audit = []
    for c in TEST_COLS:
        for per in ['pre', 'post']:
            s = df.loc[df.period == per, c]
            audit.append(dict(test=TESTS[c]['abbr'], period=per, n_records=len(s),
                              n_missing=int(s.isna().sum()), n_zero=int((s == 0).sum()),
                              n_neg=int((s < 0).sum()),
                              n_ge_cap=int((s >= TESTS[c]['cap']).sum()) if TESTS[c]['cap'] else np.nan,
                              n_gt_cap=int((s > TESTS[c]['cap']).sum()) if TESTS[c]['cap'] else np.nan,
                              n_eq_60=int((s == 60).sum()) if TESTS[c]['cap'] else np.nan,     # value heaping
                              n_eq_70=int((s == 70).sum()) if TESTS[c]['cap'] else np.nan,
                              n_eq_90=int((s == 90).sum()) if TESTS[c]['cap'] else np.nan))
        lo, hi = TESTS[c]['plausible']                  # step 0: implausible values (outside physiological limits)
        implaus = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
        for per in ['pre', 'post']:
            row = next(r for r in audit if r['test'] == TESTS[c]['abbr'] and r['period'] == per)
            row['n_implausible'] = int((implaus & (df.period == per)).sum())
        df.loc[implaus, c] = np.nan
        if rules['zero_to_na']:
            zero = df[c] == 0
            if c == 'flex':                             # 0 cm is a valid sit-and-reach score
                zero &= all_zero
            df.loc[zero, c] = np.nan
        if rules['cap'] and TESTS[c]['cap']:
            df.loc[df[c] > TESTS[c]['cap'], c] = TESTS[c]['cap']
        if rules['z_cut']:
            z = df.groupby('age')[c].transform(lambda x: (x - x.mean()) / x.std())
            out = z.abs() > rules['z_cut']
            for per in ['pre', 'post']:
                row = next(r for r in audit if r['test'] == TESTS[c]['abbr'] and r['period'] == per)
                row['n_removed_z3'] = int((out & (df.period == per)).sum())
            df.loc[out, c] = np.nan
        for per in ['pre', 'post']:
            row = next(r for r in audit if r['test'] == TESTS[c]['abbr'] and r['period'] == per)
            row['n_analysed'] = int(df.loc[df.period == per, c].notna().sum())
    flow['analytic_records'] = len(df)
    flow['pre_records']  = int((df.period == 'pre').sum())
    flow['post_records'] = int((df.period == 'post').sum())
    return df, flow, pd.DataFrame(audit)

# ----------------------------------------------------------------------------
# 2. BMI reference z-scores (KDCA 2017 primary; IOTF and WHO sensitivity)
# ----------------------------------------------------------------------------
def _lms_z(y, L, M, S):
    L = np.asarray(L); M = np.asarray(M); S = np.asarray(S); y = np.asarray(y)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(np.abs(L) < 1e-8, np.log(y / M) / S, ((y / M) ** L - 1) / (L * S))
    return z

def _interp_lms(ref, years, sex):
    """ref: DataFrame years, sex, L.bmi, M.bmi, S.bmi (half-year or monthly grid)."""
    out = np.full((len(years), 3), np.nan)
    for s in (1, 2):
        r = ref[ref.sex == s].sort_values('years')
        m = (sex == s)
        for j, col in enumerate(['L.bmi', 'M.bmi', 'S.bmi']):
            out[m, j] = np.interp(years[m], r['years'].values, r[col].values)
    return out[:, 0], out[:, 1], out[:, 2]

def add_bmi_z(df):
    years = (df['age'].values + 0.5) / 12.0          # mid-month age in years
    sex   = df['sex'].values
    bmi   = df['bmi'].values
    iotf = pd.read_csv(os.path.join(REF_DIR, 'iotf_bmi_lms.csv'))
    who  = pd.read_csv(os.path.join(REF_DIR, 'who0607_bmi_lms.csv'))
    L, M, S = _interp_lms(iotf, years, sex); df['bmiz_iotf'] = _lms_z(bmi, L, M, S)
    L, M, S = _interp_lms(who,  years, sex); df['bmiz_who']  = _lms_z(bmi, L, M, S)
    zc = pd.read_csv(os.path.join(REF_DIR, 'iotf_z_cutoffs.csv'))
    z25 = df['sex'].map(zc[zc.cut == 25].set_index('sex')['z'])
    z30 = df['sex'].map(zc[zc.cut == 30].set_index('sex')['z'])
    df['bmi_cat_iotf'] = np.select([df.bmiz_iotf >= z30, df.bmiz_iotf >= z25], ['Obese', 'Overweight'], 'Not overweight')
    df.loc[df.bmiz_iotf.isna(), 'bmi_cat_iotf'] = np.nan
    # WHO: <5 y overweight >+2 SD / obese >+3 SD ; >=5 y >+1 SD / >+2 SD
    under5 = df['age'] < 60
    ow = np.where(under5, df.bmiz_who > 2, df.bmiz_who > 1)
    ob = np.where(under5, df.bmiz_who > 3, df.bmiz_who > 2)
    df['bmi_cat_who'] = np.select([ob, ow], ['Obese', 'Overweight'], 'Not overweight')
    df.loc[df.bmiz_who.isna(), 'bmi_cat_who'] = np.nan
    kdca = os.path.join(REF_DIR, 'kdca2017_bmi_lms.csv')          # sex, month, L, M, S
    if not os.path.exists(kdca):
        raise FileNotFoundError(f'primary BMI reference not found: {kdca}')
    k = pd.read_csv(kdca).rename(columns={'month': 'years'}); k['years'] = k['years'] / 12
    k = k.rename(columns={'L': 'L.bmi', 'M': 'M.bmi', 'S': 'S.bmi'})
    L, M, S = _interp_lms(k, years, sex); df['bmiz_kdca'] = _lms_z(bmi, L, M, S)
    p = stats.norm.cdf(df['bmiz_kdca'])
    # overweight = 85th to < 95th centile, obese >= 95th; 'Not overweight' includes thinness
    df['bmi_cat_kdca'] = np.select([p >= 0.95, p >= 0.85], ['Obese', 'Overweight'], 'Not overweight')
    df.loc[df.bmiz_kdca.isna(), 'bmi_cat_kdca'] = np.nan
    # primary reference: KDCA 2017 national growth charts
    df['bmiz'] = df['bmiz_kdca']; df['bmi_cat'] = df['bmi_cat_kdca']; df.attrs['bmi_ref'] = 'KDCA 2017'
    return df

# ----------------------------------------------------------------------------
# 3. Core model: adjusted period difference
# ----------------------------------------------------------------------------
FORMULA = "{y} ~ post + female + bs(age, df=4, degree=3)"

def adjusted_difference(d, col, formula=FORMULA, extra_terms=None):
    """OLS with age spline + sex, HC3 SE. Returns dict incl. standardised deficit."""
    sub = d[[col, 'post', 'female', 'age'] + (extra_terms or [])].dropna()
    if sub['post'].nunique() < 2 or len(sub) < 50:
        return None
    f = formula.format(y=col) + (''.join(f' + {t}' for t in extra_terms) if extra_terms else '')
    m = smf.ols(f, data=sub).fit(cov_type='HC3')
    b, se = m.params['post'], m.bse['post']
    ci = m.conf_int().loc['post']
    pre  = sub.loc[sub.post == 0, col]; post = sub.loc[sub.post == 1, col]
    sd_pre = pre.std()
    sign = 1 if TESTS.get(col, {}).get('higher_better', True) else -1
    # adjusted % change relative to pre mean
    return dict(
        n_pre=len(pre), n_post=len(post),
        pre_mean=pre.mean(), pre_sd=sd_pre, post_mean=post.mean(), post_sd=post.std(),
        adj_diff=b, adj_lo=ci[0], adj_hi=ci[1], adj_se=se, p=m.pvalues['post'],
        adj_pct=100 * b / pre.mean(),
        deficit_d=-sign * b / sd_pre,
        deficit_lo=min(-sign * ci[0] / sd_pre, -sign * ci[1] / sd_pre),
        deficit_hi=max(-sign * ci[0] / sd_pre, -sign * ci[1] / sd_pre),
    )

# ----------------------------------------------------------------------------
# 4. Analyses
# ----------------------------------------------------------------------------
def table1(df):
    rows = []
    def smd(a, b):
        return abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    pre, post = df[df.post == 0], df[df.post == 1]
    for c, lab in [('age', 'Age (months)'), ('height', 'Height (cm)'), ('weight', 'Body mass (kg)'),
                   ('bmi', 'BMI (kg/m²)'), ('bmiz', f"BMI z-score ({df.attrs['bmi_ref']})"),
                   ('bmiz_iotf', 'BMI z-score (IOTF)'), ('bmiz_who', 'BMI z-score (WHO)')]:
        rows.append(dict(variable=lab, pre=f"{pre[c].mean():.2f} ± {pre[c].std():.2f}",
                         post=f"{post[c].mean():.2f} ± {post[c].std():.2f}",
                         n_pre=int(pre[c].notna().sum()), n_post=int(post[c].notna().sum()),
                         smd=round(smd(pre[c].dropna(), post[c].dropna()), 3)))
    rows.append(dict(variable='Female, n (%)', pre=f"{pre.female.sum()} ({100*pre.female.mean():.1f}%)",
                     post=f"{post.female.sum()} ({100*post.female.mean():.1f}%)",
                     n_pre=len(pre), n_post=len(post), smd=round(smd(pre.female, post.female), 3)))
    for cat_col, lab in [('bmi_cat', df.attrs['bmi_ref']), ('bmi_cat_iotf', 'IOTF'), ('bmi_cat_who', 'WHO')]:
        for cat in ['Not overweight', 'Overweight', 'Obese']:
            a = (pre[cat_col] == cat); b = (post[cat_col] == cat)
            na, nb = pre[cat_col].notna().sum(), post[cat_col].notna().sum()
            rows.append(dict(variable=f'{cat} ({lab}), n (%)', pre=f"{a.sum()} ({100*a.sum()/na:.1f}%)",
                             post=f"{b.sum()} ({100*b.sum()/nb:.1f}%)", n_pre=int(na), n_post=int(nb), smd=np.nan))
    for lo, hi in AGE_GROUPS:
        g = f'{lo}–{hi}'
        rows.append(dict(variable=f'Age {g} months, n (%)', pre=f"{(pre.age_group==g).sum()} ({100*(pre.age_group==g).mean():.1f}%)",
                         post=f"{(post.age_group==g).sum()} ({100*(post.age_group==g).mean():.1f}%)",
                         n_pre=len(pre), n_post=len(post), smd=np.nan))
    return pd.DataFrame(rows)


def overall(df):
    rows = []
    for c in TEST_COLS:
        r = adjusted_difference(df, c)
        # interpretive scale: pre-period linear change per month of age at 48-72 months (sex-adjusted);
        # months_equivalent = adjusted difference / slope (negative = behind the pre-period age norm)
        pre = df[(df.post == 0) & df.age.between(48, 72)][[c, 'age', 'female']].dropna()
        slope = smf.ols(f'{c} ~ age + female', data=pre).fit().params['age']
        r.update(test=TESTS[c]['abbr'], label=label(c), higher_better=TESTS[c]['higher_better'],
                 age_slope_48_72=slope, months_equivalent=r['adj_diff'] / slope)
        rows.append(r)
    cols = ['test', 'label', 'n_pre', 'n_post', 'pre_mean', 'pre_sd', 'post_mean', 'post_sd',
            'adj_diff', 'adj_lo', 'adj_hi', 'adj_pct', 'deficit_d', 'deficit_lo', 'deficit_hi', 'p',
            'age_slope_48_72', 'months_equivalent']
    return pd.DataFrame(rows)[cols]


def by_age_group(df):
    rows = []
    for c in TEST_COLS:
        for lo, hi in AGE_GROUPS:
            g = f'{lo}–{hi}'
            r = adjusted_difference(df[df.age_group == g], c, formula="{y} ~ post + female + age")
            if r is None: continue
            r.update(test=TESTS[c]['abbr'], age_group=g)
            rows.append(r)
    # interaction test: period x age group (joint Wald)
    inter = []
    for c in TEST_COLS:
        sub = df[[c, 'post', 'female', 'age', 'age_group']].dropna()
        m = smf.ols(f"{c} ~ post*C(age_group) + female + bs(age, df=4, degree=3)", data=sub).fit(cov_type='HC3')
        names = [n for n in m.params.index if n.startswith('post:')]
        w = m.wald_test(' = '.join(names) + ' = 0' if len(names) == 1 else ', '.join(f'{n} = 0' for n in names), scalar=True)
        inter.append(dict(test=TESTS[c]['abbr'], wald_chi2=float(w.statistic), df=len(names), p_interaction=float(w.pvalue)))
    return pd.DataFrame(rows), pd.DataFrame(inter)


def by_sex_bmi(df, cat_col='bmi_cat'):
    rows = []
    for c in TEST_COLS:
        for sex, sname in [(1, 'Male'), (2, 'Female')]:
            for cat in ['Not overweight', 'Overweight', 'Obese']:
                sub = df[(df.sex == sex) & (df[cat_col] == cat)]
                r = adjusted_difference(sub, c, formula="{y} ~ post + bs(age, df=4, degree=3)")
                if r is None: continue
                r.update(test=TESTS[c]['abbr'], sex=sname, bmi_cat=cat, reference=cat_col)
                rows.append(r)
    return pd.DataFrame(rows)


def bmi_change(df):
    rows = []
    for sex, sname in [(1, 'Male'), (2, 'Female')]:
        for lo, hi in AGE_GROUPS:
            g = f'{lo}–{hi}'
            sub = df[(df.sex == sex) & (df.age_group == g)]
            for y in ['bmi', 'bmiz']:
                r = adjusted_difference(sub, y, formula="{y} ~ post + age")
                if r is None: continue
                r.update(outcome=y, sex=sname, age_group=g)
                rows.append(r)
    for y in ['bmi', 'bmiz']:
        r = adjusted_difference(df, y); r.update(outcome=y, sex='All', age_group='36–84'); rows.append(r)
    return pd.DataFrame(rows)[['outcome', 'sex', 'age_group', 'n_pre', 'n_post', 'pre_mean', 'pre_sd', 'post_mean', 'post_sd',
                               'adj_diff', 'adj_lo', 'adj_hi', 'p']]


def yearly_trend(df):
    """Age-sex-standardised annual means. One model per test with calendar year as a categorical
    term, y ~ C(year) + female + bs(age), HC3 covariance; for each year the predicted values of all
    records (pooled sample, i.e. a common age-sex distribution) are averaged with the year set to
    that year. SE by the delta method. The pre-pandemic slope is a straight line through the three
    2017-2019 standardised means (descriptive only)."""
    rows = []
    for c in TEST_COLS:
        sub = df[[c, 'female', 'age', 'year']].dropna().copy()
        sub['year'] = sub['year'].astype(int)
        m = smf.ols(f"{c} ~ C(year) + female + bs(age, df=4, degree=3)", data=sub).fit(cov_type='HC3')
        V = m.cov_params().values; b = m.params.values
        years = sorted(sub.year.unique()); est = {}
        for y in years:
            X = build_design_matrices([m.model.data.design_info], sub.assign(year=y), return_type='dataframe')[0]
            g = X.mean(axis=0).values
            est[y] = (g @ b, float(np.sqrt(g @ V @ g)))
        n = sub.groupby('year').size()
        pre = np.array([est[y][0] for y in (2017, 2018, 2019)])
        slope, intercept = np.polyfit([2017, 2018, 2019], pre, 1)
        for y in years:
            rows.append(dict(test=TESTS[c]['abbr'], year=y, adj_mean=est[y][0], se=est[y][1], n=int(n[y]),
                             pre_slope_per_year=slope,
                             pre_trend_extrapolated=intercept + slope * y if y >= 2022 else np.nan))
    return pd.DataFrame(rows)


# 6-month age bands (the last band, 78-84, spans 7 months)
BANDS = [(lo, lo + 5) for lo in range(AGE_MIN, 78, 6)] + [(78, AGE_MAX)]


def empirical_percentiles(df, tests=TEST_COLS, qs=(10, 50, 90)):
    """Percentiles by 6-month age band and period (sexes pooled; Supplementary Table S4A)."""
    rows = []
    for c in tests:
        for lo, hi in BANDS:
            sub = df[(df.age >= lo) & (df.age <= hi)]
            pre = sub.loc[sub.post == 0, c].dropna().values; post = sub.loc[sub.post == 1, c].dropna().values
            qp = np.percentile(pre, qs); qo = np.percentile(post, qs)
            for i, q in enumerate(qs):
                rows.append(dict(test=TESTS[c]['abbr'], age_bin=f'{lo}–{hi}', age_mid=(lo + hi) / 2,
                                 n_pre=len(pre), n_post=len(post), percentile=q,
                                 pre=qp[i], post=qo[i], diff=qo[i] - qp[i]))
    return pd.DataFrame(rows)


def band_percentile_summary(df, tests=TEST_COLS, qs=(10, 25, 50, 75, 90), n_boot=2000, seed=42):
    """Primary distributional analysis (Table 4). Within each 6-month age band, the post-pre
    difference in each percentile is computed (sexes pooled); differences are averaged across bands,
    weighted by the number of records in the band. Percent differences are averaged in the same way
    from the band-specific percent differences. 95 % CIs: percentile bootstrap with resampling of
    records within band and period."""
    r = np.random.default_rng(seed)
    rows = []
    for c in tests:
        w, pre_q, post_q, bpre, bpost = [], [], [], [], []
        for lo, hi in BANDS:
            sub = df[(df.age >= lo) & (df.age <= hi)]
            a = sub.loc[sub.post == 0, c].dropna().values; b = sub.loc[sub.post == 1, c].dropna().values
            if len(a) == 0 or len(b) == 0:
                continue
            w.append(len(a) + len(b))
            pre_q.append(np.percentile(a, qs)); post_q.append(np.percentile(b, qs))
            bpre.append(np.percentile(r.choice(a, (n_boot, len(a))), qs, axis=1).T)    # n_boot x len(qs)
            bpost.append(np.percentile(r.choice(b, (n_boot, len(b))), qs, axis=1).T)
        w = np.array(w, float) / sum(w)
        pre_q, post_q = np.array(pre_q), np.array(post_q)                           # bands x qs
        bpre, bpost = np.array(bpre), np.array(bpost)                               # bands x n_boot x qs
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = 100 * (post_q - pre_q) / pre_q
            bpct = 100 * (bpost - bpre) / bpre
        diff = w @ (post_q - pre_q); bdiff = np.einsum('b,bkq->kq', w, bpost - bpre)
        pct_avg = w @ np.nan_to_num(pct); bpct_avg = np.einsum('b,bkq->kq', w, np.nan_to_num(bpct))
        for i, q in enumerate(qs):
            rows.append(dict(test=TESTS[c]['abbr'], percentile=q, n_bands=len(w), pre_value=w @ pre_q[:, i], post_value=w @ post_q[:, i],
                             diff=diff[i], diff_lo=np.percentile(bdiff[:, i], 2.5), diff_hi=np.percentile(bdiff[:, i], 97.5),
                             pct=pct_avg[i], pct_lo=np.percentile(bpct_avg[:, i], 2.5), pct_hi=np.percentile(bpct_avg[:, i], 97.5)))
    return pd.DataFrame(rows)


def low_range_share(df, tests=TEST_COLS, n_boot=2000, seed=7):
    """Share of records in the low fitness range of the pre-pandemic distribution.
    Within each 6-month age band the pre-pandemic 10th fitness percentile is the cut-off (for the
    5SR, where a longer time is worse, the 90th percentile of time).
    Band-specific shares are combined with the same weights in both periods (records in the band, both periods pooled), so that a change in
    the age composition does not enter the comparison. Values equal to a cut-off are not counted as
    beyond it, so the pre-pandemic share can be below 10 % for tests with heaped values. 95 % CI for
    the post-pre difference (percentage points): bootstrap of records within band and period, with
    the cut-offs re-estimated in each resample."""
    r = np.random.default_rng(seed)

    def shares(a, b, hb):
        p10, p90 = np.percentile(a, [10, 90])
        if hb:
            return np.array([(a < p10).mean(), (b < p10).mean()])
        return np.array([(a > p90).mean(), (b > p90).mean()])

    rows = []
    for c in tests:
        hb = TESTS[c]['higher_better']
        parts, boots, w = [], [], []
        for lo, hi in BANDS:
            sub = df[(df.age >= lo) & (df.age <= hi)]
            a = sub.loc[sub.post == 0, c].dropna().values; b = sub.loc[sub.post == 1, c].dropna().values
            if len(a) == 0 or len(b) == 0:
                continue
            w.append(len(a) + len(b)); parts.append(shares(a, b, hb))
            boots.append(np.array([shares(r.choice(a, len(a)), r.choice(b, len(b)), hb) for _ in range(n_boot)]))
        w = np.array(w, float) / sum(w)
        est = 100 * w @ np.array(parts)                              # [pre_low, post_low]
        bs = 100 * np.einsum('b,bkq->kq', w, np.array(boots))
        d_low = bs[:, 1] - bs[:, 0]
        rows.append(dict(test=TESTS[c]['abbr'], pre_low=est[0], post_low=est[1], diff_low=est[1] - est[0],
                         diff_low_lo=np.percentile(d_low, 2.5), diff_low_hi=np.percentile(d_low, 97.5)))
    return pd.DataFrame(rows)


def band_sensitivity(df_raw, tests=('endu', 'equi'), qs=(50, 90)):
    """Band-averaged percentile differences for VSE/SLS under alternative cleaning rules. Each rule is run
    with the same call as Table 4 (all tests, same seed), so the primary rows reproduce Table 4 exactly."""
    specs = {'primary': dict(zero_to_na=True, cap=True, z_cut=3.0),
             'uncapped': dict(zero_to_na=True, cap=False, z_cut=3.0), 'no_z_trim': dict(zero_to_na=True, cap=True, z_cut=None)}
    out = []
    for name, rule in specs.items():
        d, _, _ = clean(df_raw, rule)
        t = band_percentile_summary(d)
        t = t[t.test.isin([TESTS[c]['abbr'] for c in tests]) & t.percentile.isin(qs)]
        t.insert(0, 'spec', name); out.append(t)
    return pd.concat(out, ignore_index=True)


def to_performance_percentiles(d, col='percentile', top=100):
    """For tests in which a lower value is better (the 5SR), report percentiles of the time distribution as
    fitness percentiles: the q-th percentile of time becomes the (top - q)-th fitness percentile, so
    that for every test low percentiles denote low fitness (the 10th = the slowest tenth for the 5SR).
    Only the label changes; values are unchanged. The percentile sets used are symmetric."""
    d = d.copy()
    lower_better = d.test.isin([TESTS[c]['abbr'] for c in TEST_COLS if not TESTS[c]['higher_better']])
    d.loc[lower_better, col] = (top - d.loc[lower_better, col]).round(6)
    order = {TESTS[c]['abbr']: i for i, c in enumerate(TEST_COLS)}
    keys = ['_o'] + (['age_mid'] if 'age_mid' in d.columns else []) + [col]
    return d.assign(_o=d.test.map(order)).sort_values(keys, kind='stable').drop(columns='_o').reset_index(drop=True)


def add_deficit(d, est, lo, hi, name):
    """Add the difference oriented so that a positive value denotes a post-pandemic deficit, as for the
    standardised differences: -(post - pre) for tests in which a higher value is better, +(post - pre) for the 5SR."""
    sign = d.test.map({TESTS[c]['abbr']: (-1 if TESTS[c]['higher_better'] else 1) for c in TEST_COLS})
    a, b = sign * d[lo], sign * d[hi]
    return d.assign(**{name: sign * d[est], f'{name}_lo': np.minimum(a, b), f'{name}_hi': np.maximum(a, b)})


def deficit_columns(d):
    """Table 4 / Table S4: band-averaged differences (original units and %) oriented so that + = deficit."""
    d = add_deficit(d, 'diff', 'diff_lo', 'diff_hi', 'deficit')
    return add_deficit(d, 'pct', 'pct_lo', 'pct_hi', 'deficit_pct')


def quantile_regression(df, taus=(0.10, 0.25, 0.50, 0.75, 0.90)):
    """Adjusted period difference (original units) at each quantile, with the primary-model covariates."""
    rows = []
    for c in TEST_COLS:
        sub = df[[c, 'post', 'female', 'age']].dropna()
        y, X = dmatrices(FORMULA.format(y=c), sub, return_type='dataframe')
        for tau in taus:
            m = sm.QuantReg(y, X).fit(q=tau, max_iter=5000)
            ci = m.conf_int().loc['post']
            rows.append(dict(test=TESTS[c]['abbr'], tau=tau, coef=m.params['post'], lo=ci[0], hi=ci[1],
                             p=m.pvalues['post'], n=len(sub)))
    return pd.DataFrame(rows)


def sex_bmi_interactions(df):
    """Period x sex and period x BMI-category interactions (HC3), one model per test, BMI-complete records."""
    rows = []
    for c in TEST_COLS:
        sub = df[[c, 'post', 'female', 'age', 'bmi_cat']].dropna()
        m = smf.ols(f"{c} ~ post*female + post*C(bmi_cat) + bs(age, df=4, degree=3)", data=sub).fit(cov_type='HC3')
        bmi_terms = [x for x in m.params.index if x.startswith('post:C(bmi_cat)')]
        w = m.wald_test(', '.join(f'{x} = 0' for x in bmi_terms), scalar=True)
        rows.append(dict(test=TESTS[c]['abbr'], n=len(sub), p_period_x_sex=float(m.pvalues['post:female']),
                         coef_period_x_female=float(m.params['post:female']), p_period_x_bmi=float(w.pvalue)))
    return pd.DataFrame(rows)


def _value_key(d, cols, zero_as_missing=()):
    """Exact-value linkage key: fields rounded to 2 decimals and formatted identically in both files
    (missing -> -999; zeros of the listed fields treated as missing)."""
    v = d[cols].apply(pd.to_numeric, errors='coerce').astype(float).round(2)
    for c in zero_as_missing:
        v[c] = v[c].where(v[c] != 0, np.nan)
    return v.fillna(-999).map(lambda x: f'{x:.2f}').agg('|'.join, axis=1)


def restore_pre_srt_zeros(df_raw):
    """In the 2017-2019 records of the analysis file all zero values had been set to missing during data
    preparation. Sit-and-reach zeros are valid scores, so they are restored from the 2017-2019 records of the
    earlier report's input file (PFN_PRIOR), in which zeros were kept:
    a present record with a missing sit-and-reach value is set to 0 when it links to an original record with
    sit-and-reach 0 and non-zero values for the other four tests (zeros in all tests mark records without
    measurements). Links require identical year, sex, age, height, body mass, standing long jump and shuttle-run
    values (a combination occurring once in each file). Returns (data, counts) or (data, None)."""
    if not os.path.exists(PRIOR_PATH):
        return df_raw, None
    o = pd.read_csv(PRIOR_PATH, encoding='utf-8-sig', low_memory=False)
    for c in o.columns:
        o[c] = pd.to_numeric(o[c], errors='coerce')
    o = o[o.year.between(2017, 2019)]
    K = ['year', 'sex', 'age', 'height', 'weight', 'quick', 'agil']
    pre = df_raw[df_raw.year <= 2019].copy()
    pre['key'] = _value_key(pre, K, ('quick', 'agil')); o['key'] = _value_key(o, K, ('quick', 'agil'))
    genuine = (o.flex == 0) & ((o[['endu', 'quick', 'equi', 'agil']] == 0).sum(axis=1) == 0)
    o_unique = set(o.key[~o.key.duplicated(keep=False)])
    pre_unique = pre[~pre.key.duplicated(keep=False)]
    hit = pre_unique.flex.isna() & pre_unique.key.isin(set(o.key[genuine]) & o_unique)
    out = df_raw.copy(); out.loc[pre_unique.index[hit], 'flex'] = 0.0
    counts = dict(original_pre_records=int(len(o)), original_srt_zero=int((o.flex == 0).sum()),
                  original_all_tests_zero=int((o[TEST_COLS] == 0).all(axis=1).sum()),
                  original_srt_zero_genuine=int(genuine.sum()), restored=int(hit.sum()))
    return out, counts


def pre_repeat_flags(df_raw):
    """Identify 2017-2019 records that are a second or later measurement of the same child, by
    linking to the programme's repeat-measurement file (one sheet per number of measurements,
    one row per child, measurements side by side). The two files were prepared from different
    versions of the programme data, so values can differ slightly (zeros set to missing, last digits).
    A candidate pair has the same sex, height and body mass within 0.5 (cm, kg), age within 3 months,
    and agreement on at least four of the five test values (values within 0.1, or both zero or
    missing). Each record is linked at most once: pairs are taken in order of test agreement, then
    of closeness in age, height and body mass. Records linked to a second or later measurement are
    flagged. Optional; returns (index of flagged records, aggregate counts) or (empty index, None)."""
    if not REPEAT_PATH or not os.path.exists(REPEAT_PATH):
        return pd.Index([]), None
    base = ['Birth', '성별', '신장(cm)', '체중(kg)', '유연성(cm)', '복부지구력(초)', '순발력(cm)', '평형성(초)', '민첩성(초)']
    names = ['age', 'sex', 'height', 'weight', 'flex', 'endu', 'quick', 'equi', 'agil']
    parts = []
    for sheet, t in pd.read_excel(REPEAT_PATH, sheet_name=None).items():
        k = 0
        while all((c + (f'.{k}' if k else '')) in t.columns for c in base):
            p = t[['ID'] + [c + (f'.{k}' if k else '') for c in base]].copy()
            p.columns = ['ID'] + names; p['k'] = k; parts.append(p); k += 1
    rep = pd.concat(parts, ignore_index=True)
    for c in names:
        rep[c] = pd.to_numeric(rep[c], errors='coerce')
    rep = rep.dropna(subset=['sex', 'height']).reset_index(drop=True)
    rep['rid'] = rep.index
    T = ['flex', 'endu', 'quick', 'equi', 'agil']
    pre = df_raw[df_raw.year <= 2019]
    P = pre.loc[pre.height.notna(), ['sex', 'age', 'height', 'weight'] + T].copy()
    P['pid'] = P.index
    for d in (rep, P):
        d['sex'] = d['sex'].astype(float)
    P['hb'] = np.floor(P.height).astype(int)                     # blocking on height (+/- 1 cm band)
    R = pd.concat([rep.assign(hb=np.floor(rep.height).astype(int) + d) for d in (-1, 0, 1)], ignore_index=True)
    m = R.merge(P, on=['sex', 'hb'], suffixes=('_r', '_p'))
    m = m[((m.age_r - m.age_p).abs() <= 3) & ((m.height_r - m.height_p).abs() <= 0.5 + 1e-9)
          & ((m.weight_r - m.weight_p).abs() <= 0.5 + 1e-9)].copy()

    def agree(x, y):
        return ((x.isna() | (x == 0)) & (y.isna() | (y == 0))) | ((x - y).abs() <= 0.1 + 1e-9)
    m['score'] = sum(agree(m[c + '_r'], m[c + '_p']).astype(int) for c in T)
    m['dist'] = (m.age_r - m.age_p).abs() + (m.height_r - m.height_p).abs() + (m.weight_r - m.weight_p).abs()
    m = m[m.score >= 4].sort_values(['score', 'dist', 'rid', 'pid'], ascending=[False, True, True, True], kind='mergesort')
    used_r, used_p, linked = set(), set(), []
    for rid, pid, k in zip(m.rid, m.pid, m.k):
        if rid in used_r or pid in used_p:
            continue
        used_r.add(rid); used_p.add(pid); linked.append((pid, k))
    L = pd.DataFrame(linked, columns=['pid', 'k'])
    flagged = pd.Index(sorted(L.pid[L.k >= 1]))
    s = rep.sort_values(['ID', 'k'])
    gap = s.groupby('ID').age.diff()[s.k == 1].dropna()          # age difference (months), first to second measurement
    counts = dict(repeat_file_children=int(rep.ID.nunique()), repeat_file_records=int(len(rep)),
                  repeat_file_later_measurements=int((rep.k >= 1).sum()), pre_records=int(len(pre)),
                  pre_linked=int(len(L)), pre_linked_later_measurement=int(len(flagged)),
                  interval_1to2_median=float(gap.median()), interval_1to2_pct_3to7=round(100 * gap.between(3, 7).mean(), 1))
    return flagged, counts


def fig_flow(flow, audit, df):
    """Supplementary Figure S1: participant flow (records)."""
    fig, ax = plt.subplots(figsize=(8.2, 6.5)); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def box(x, y, text, w=0.42, h=0.09):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=False, lw=1))
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5)
    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=0.8))
    box(0.5, 0.92, f"Records in the analysis file, 2017–2024\n$n$ = {flow['raw_records']:,}\n(records lacking age or sex removed when the file was prepared)", w=0.62, h=0.11)
    n_inv = flow['raw_records'] - flow['after_missing_sex_age']
    n_out = flow['after_missing_sex_age'] - flow['after_age_36_84']
    arrow(0.5, 0.865, 0.5, 0.655); arrow(0.5, 0.765, 0.66, 0.765)
    box(0.83, 0.765, f"Excluded, $n$ = {n_inv + n_out:,}\nAge not computable: {n_inv:,}\nOutside 36–84 months: {n_out:,}", w=0.32, h=0.11)
    box(0.5, 0.61, f"Aged 36–84 months\n$n$ = {flow['after_age_36_84']:,}", w=0.56)
    arrow(0.5, 0.565, 0.22, 0.475); arrow(0.5, 0.565, 0.78, 0.475)
    npre, npost = flow['pre_records'], flow['post_records']
    box(0.22, 0.43, f"Pre-pandemic 2017–2019\n$n$ = {npre:,}", w=0.36); box(0.78, 0.43, f"Post-pandemic 2022–2024\n$n$ = {npost:,}", w=0.36)
    bmi_pre = int(df[(df.post == 0)].bmi.notna().sum()); bmi_post = int(df[(df.post == 1)].bmi.notna().sum())
    lines_pre = [f"BMI: {bmi_pre:,}"]; lines_post = [f"BMI: {bmi_post:,}"]
    for c in TEST_COLS:
        a = audit[(audit.test == TESTS[c]['abbr'])].set_index('period')
        lines_pre.append(f"{TESTS[c]['abbr']}: {int(a.loc['pre', 'n_analysed']):,}"); lines_post.append(f"{TESTS[c]['abbr']}: {int(a.loc['post', 'n_analysed']):,}")
    arrow(0.22, 0.385, 0.22, 0.3); arrow(0.78, 0.385, 0.78, 0.3)
    box(0.22, 0.17, "Records analysed per outcome, $n$\n" + "\n".join(lines_pre), w=0.36, h=0.24); box(0.78, 0.17, "Records analysed per outcome, $n$\n" + "\n".join(lines_post), w=0.36, h=0.24)
    fig.savefig(os.path.join(FIG, 'figS1_flow.png'), dpi=300, bbox_inches='tight'); plt.close(fig)


# Published V-sit endurance percentiles of Oh & Lee (2025), Children 12:361, Table 3 (GAMLSS; s)
PUBLISHED_VSE = {  # (sex, age in months): (P10, P50, P90)
    (1, 36): (0.6, 3.0, 8.6),  (1, 48): (1.6, 9.1, 28.2),  (1, 60): (4.2, 21.9, 65.8),
    (1, 72): (7.9, 33.7, 91.6), (1, 84): (12.9, 44.0, 106.1),
    (2, 36): (0.7, 3.7, 10.8), (2, 48): (2.2, 11.6, 34.4), (2, 60): (5.5, 26.7, 77.4),
    (2, 72): (10.0, 38.4, 98.9), (2, 84): (16.0, 50.3, 116.3),
}


def vse_empirical_vs_published(df):
    """Supplementary Table S7: empirical V-sit percentiles of the present 2017-2019 records
    (records within +/- 1 month of each target age, by sex) next to the published values."""
    pre = df[(df.post == 0) & df.endu.notna()]
    rows = []
    for (sex, age), pub in PUBLISHED_VSE.items():
        v = pre[(pre.sex == sex) & ((pre.age - age).abs() <= 1)].endu.values
        q = np.percentile(v, [10, 50, 90]) if len(v) else [np.nan] * 3
        rows.append(dict(sex=sex, age=age, n=len(v), p10_empirical=q[0], p50_empirical=q[1], p90_empirical=q[2],
                         p10_published=pub[0], p50_published=pub[1], p90_published=pub[2]))
    return pd.DataFrame(rows)


def robustness(df_raw, df_primary, pre_repeats=pd.Index([])):
    """Standardised deficit under alternative specifications."""
    out = {}
    base = overall(df_primary).set_index('test')['deficit_d']; out['primary'] = base
    # BMI-complete subset, without and with BMI z-score (primary BMI reference) as covariate
    bmi_sub = df_primary[df_primary.bmiz.notna()]
    out['primary_bmi_subset'] = pd.Series({TESTS[c]['abbr']: adjusted_difference(bmi_sub, c)['deficit_d'] for c in TEST_COLS})
    out['plus_bmiz'] = pd.Series({TESTS[c]['abbr']: adjusted_difference(df_primary, c, extra_terms=['bmiz'])['deficit_d'] for c in TEST_COLS})
    # first measurements only: post-pandemic records with index 1; pre-pandemic records without
    # those identified as a later measurement of the same child (pre_repeat_flags), adjusted
    first = df_primary[((df_primary.post == 1) & (df_primary.time == 1)) |
                       ((df_primary.post == 0) & ~df_primary.index.isin(pre_repeats))]
    out['first_only_adj'] = pd.Series({TESTS[c]['abbr']: adjusted_difference(first, c)['deficit_d'] for c in TEST_COLS})
    print(f'   first measurements only: {int((first.post == 0).sum())} pre / {int((first.post == 1).sum())} post records')
    # no cap
    d1, _, _ = clean(df_raw, dict(zero_to_na=True, cap=False, z_cut=3.0))
    out['uncapped'] = pd.Series({TESTS[c]['abbr']: adjusted_difference(d1, c)['deficit_d'] for c in TEST_COLS})
    # no outlier trimming (zeros->NA, cap applied)
    d2, _, _ = clean(df_raw, dict(zero_to_na=True, cap=True, z_cut=None))
    out['no_z_trim'] = pd.Series({TESTS[c]['abbr']: adjusted_difference(d2, c)['deficit_d'] for c in TEST_COLS})
    return pd.DataFrame(out)


def repeat_advantage(df, pre_repeats):
    """Age-sex-adjusted difference between later and first measurements within each period, in
    pre-period SD (positive = later measurements perform better). Post: index > 1 vs index 1;
    pre: records identified as later measurements (pre_repeat_flags) vs all others."""
    rows = []
    for c in TEST_COLS:
        sd = df.loc[df.post == 0, c].std(); sign = 1 if TESTS[c]['higher_better'] else -1
        for p, lab in ((0, 'pre'), (1, 'post')):
            d = df[df.post == p].copy()
            d['later'] = (d.time > 1).astype(int) if p == 1 else d.index.isin(pre_repeats).astype(int)
            s = d[[c, 'later', 'female', 'age']].dropna()
            if s.later.nunique() < 2:
                continue
            m = smf.ols(f'{c} ~ later + female + bs(age, df=4, degree=3)', data=s).fit(cov_type='HC3')
            rows.append(dict(test=TESTS[c]['abbr'], period=lab, n_later=int(s.later.sum()), n=len(s),
                             diff_sd=sign * m.params['later'] / sd))
    return pd.DataFrame(rows)


HARMONISED_DECIMALS = {'endu': 0, 'equi': 0, 'flex': 0, 'quick': 0, 'agil': 1}

def recording_checks(df):
    """Recording format by period (share of whole numbers and of values with two or more decimals)
    and the standardised difference (i) with all values rounded to a common precision in both
    periods and (ii) with VSE and SLS capped at 60 s in both periods."""
    rows = []
    for c in TEST_COLS:
        r = dict(test=TESTS[c]['abbr'])
        for p, lab in ((0, 'pre'), (1, 'post')):
            v = df.loc[df.post == p, c].dropna()
            r[f'pct_whole_{lab}'] = 100 * np.isclose(v, v.round(0)).mean()
            r[f'pct_2dec_{lab}'] = 100 * (1 - np.isclose(v, v.round(1)).mean())
        r['primary'] = adjusted_difference(df, c)['deficit_d']
        h = df.copy(); h[c] = h[c].round(HARMONISED_DECIMALS[c])
        r['harmonised_precision'] = adjusted_difference(h, c)['deficit_d']
        if c in ('endu', 'equi'):
            k = df.copy(); k[c] = k[c].clip(upper=60)
            r['cap_60s'] = adjusted_difference(k, c)['deficit_d']
        else:
            r['cap_60s'] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def design_effect(df, ov, iccs=(0.1, 0.3, 0.5, 0.7)):
    """Illustrative clustering scenario (not an upper bound): primary SEs multiplied by
    sqrt(1 + (m_bar - 1) * ICC), m_bar = post-period records per index-1 record. Pre-period
    repeats, unequal cluster sizes and centre-level clustering are not represented."""
    post = df[df.post == 1]
    m_bar = len(post) / (post.time == 1).sum()
    rows = []
    for _, r in ov.iterrows():
        se = (r.deficit_hi - r.deficit_lo) / (2 * 1.96)
        for icc in iccs:
            f = np.sqrt(1 + (m_bar - 1) * icc)
            rows.append(dict(test=r.test, m_bar=m_bar, icc=icc, se_inflation=f, deficit_d=r.deficit_d,
                             lo=r.deficit_d - 1.96 * se * f, hi=r.deficit_d + 1.96 * se * f))
    return pd.DataFrame(rows)


def missingness(df):
    """Is missing a test associated with fitness on the other tests? Composite age-specific z of
    the available other tests, by missing status and period."""
    z = pd.DataFrame(index=df.index)
    for c in TEST_COLS:
        zz = df.groupby('age')[c].transform(lambda x: (x - x.mean()) / x.std())
        z[c] = zz if TESTS[c]['higher_better'] else -zz
    rows = []
    for c in TEST_COLS:
        others = [o for o in TEST_COLS if o != c]
        comp = z[others].mean(axis=1)
        for per in ['pre', 'post']:
            m = df.period == per
            miss = df[c].isna() & m; have = df[c].notna() & m
            rows.append(dict(test=TESTS[c]['abbr'], period=per,
                             n_missing=int(miss.sum()), pct_missing=100 * miss.sum() / m.sum(),
                             composite_z_missing=comp[miss].mean(), composite_z_present=comp[have].mean(),
                             n_missing_with_other_tests=int(comp[miss].notna().sum())))
    return pd.DataFrame(rows)


def repeat_structure(df):
    t = pd.crosstab(df.year, df.time)
    t['records'] = t.sum(axis=1)
    t['pct_time_ge2'] = 100 * (t['records'] - t.get(1, 0)) / t['records']
    g = df.groupby('year').agg(age_mean=('age', 'mean'), age_sd=('age', 'std'), pct_female=('female', 'mean'))
    g['pct_female'] *= 100
    return t.join(g.round(2))

# ----------------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------------
COLORS = {'VSE': '#1f77b4', 'SRT': '#2ca02c', 'SLS': '#9467bd', 'SLJ': '#ff7f0e', '5SR': '#d62728'}

def fig_yearly(tr):
    fig, axes = plt.subplots(1, 5, figsize=(10, 3.3))
    for ax, c in zip(axes, TEST_COLS):
        a = TESTS[c]['abbr']; g = tr[tr.test == a]
        pre = g[g.year <= 2019]; post = g[g.year >= 2022]
        for seg, ls in [(pre, '-'), (post, '--')]:
            ax.errorbar(seg.year, seg.adj_mean, yerr=1.96 * seg.se, fmt='o', ls=ls, color=COLORS[a], capsize=2, ms=4)
        ax.axvspan(2019.5, 2021.5, color='grey', alpha=0.15)
        ax.set_title(f"{a} ({TESTS[c]['unit']})" + (", axis inverted" if not TESTS[c]['higher_better'] else ""), fontsize=10.5)
        ax.set_xticks([2017, 2018, 2019, 2022, 2023, 2024])
        ax.set_xticklabels(['2017', '2018', '2019', '2022', '2023', '2024'], rotation=45, ha='right', rotation_mode='anchor')
        ax.tick_params(labelsize=9)
        if not TESTS[c]['higher_better']:
            ax.invert_yaxis()
    axes[0].set_ylabel('Age–sex-standardised\nmean', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_yearly_trend.png'), dpi=600); plt.close(fig)


def fig_age(ag):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(AGE_GROUPS)); w = 0.15
    for i, c in enumerate(TEST_COLS):
        a = TESTS[c]['abbr']; g = ag[ag.test == a].set_index('age_group').reindex([f'{lo}–{hi}' for lo, hi in AGE_GROUPS])
        ax.bar(x + (i - 2) * w, g.deficit_d, w, yerr=[g.deficit_d - g.deficit_lo, g.deficit_hi - g.deficit_d],
               color=COLORS[a], label=a, capsize=2, error_kw=dict(lw=0.8))
    ax.axhline(0, color='k', lw=0.6); ax.axhline(0.2, color='grey', lw=0.6, ls=':')
    ax.set_xticks(x); ax.set_xticklabels([f'{lo}–{hi} mo' for lo, hi in AGE_GROUPS])
    ax.set_ylabel('Standardised post-pandemic deficit\n(adjusted difference / pre-period SD; + = worse)')
    ax.legend(ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_age_gradient.png'), dpi=600); plt.close(fig)


def fig_quantile(bp):
    """Figure 3. Band-averaged post-pandemic deficit (%) at the 10th-90th fitness percentiles, with bootstrap CIs."""
    fig, axes = plt.subplots(1, 5, figsize=(10, 3.2))
    for j, c in enumerate(TEST_COLS):
        a = TESTS[c]['abbr']
        g = bp[bp.test == a]
        y, lo, hi = g.deficit_pct, g.deficit_pct_lo, g.deficit_pct_hi      # positive = worse for every test
        ax = axes[j]
        ax.errorbar(g.percentile, y, yerr=[y - lo, hi - y], fmt='o-', color=COLORS[a], capsize=2)
        ax.axhline(0, color='k', lw=0.5); ax.set_xticks([10, 50, 90]); ax.set_title(a, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_xlabel('Fitness percentile', fontsize=9.5); ax.set_ylabel('Post-pandemic deficit (%)\n(+ = worse)' if j == 0 else '', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_distribution.png'), dpi=600); plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    raw0 = load_raw()
    raw, srt_counts = restore_pre_srt_zeros(raw0)
    if srt_counts is None:
        print('(input file of the earlier report not found: pre-pandemic sit-and-reach zeros are not restored)')
    else:
        pd.DataFrame([srt_counts]).to_csv(os.path.join(SUP, 'S_srt_zero_restore.csv'), index=False)
        print("Sit-and-reach zeros restored from the earlier report's input file:", srt_counts)
    df, flow, audit = clean(raw)
    df = add_bmi_z(df)
    print('FLOW:', json.dumps(flow, indent=1))
    with open(os.path.join(TAB, 'flow.json'), 'w', encoding='utf-8') as fh:
        json.dump(flow, fh, indent=1)
    audit.to_csv(os.path.join(SUP, 'S_audit_cleaning.csv'), index=False)
    print('\n== Cleaning audit ==\n', audit.to_string(index=False))

    t1 = table1(df); t1.to_csv(os.path.join(TAB, 'table1_baseline.csv'), index=False)
    print('\n== Table 1 ==\n', t1.to_string(index=False))

    ov = overall(df); ov.to_csv(os.path.join(TAB, 'table2_overall.csv'), index=False)
    print('\n== Table 2: overall adjusted differences ==')
    print(ov[['test', 'n_pre', 'n_post', 'pre_mean', 'post_mean', 'adj_diff', 'adj_lo', 'adj_hi', 'adj_pct',
              'deficit_d', 'deficit_lo', 'deficit_hi', 'p']].round(3).to_string(index=False))

    ag, inter = by_age_group(df)
    ag.to_csv(os.path.join(TAB, 'table3_age_group.csv'), index=False); inter.to_csv(os.path.join(TAB, 'table3_interaction.csv'), index=False)
    print('\n== Table 3: by age group ==')
    print(ag[['test', 'age_group', 'n_pre', 'n_post', 'pre_mean', 'post_mean', 'adj_diff', 'adj_pct', 'deficit_d', 'deficit_lo', 'deficit_hi', 'p']].round(3).to_string(index=False))
    print(inter.round(4).to_string(index=False))

    sb = by_sex_bmi(df); sb.to_csv(os.path.join(SUP, 'S_sex_bmi_primary.csv'), index=False)
    by_sex_bmi(df, 'bmi_cat_iotf').to_csv(os.path.join(SUP, 'S_sex_bmi_iotf.csv'), index=False)
    by_sex_bmi(df, 'bmi_cat_who').to_csv(os.path.join(SUP, 'S_sex_bmi_who.csv'), index=False)
    print('\n== Sex x BMI ({}), VSE & 5SR =='.format(df.attrs.get('bmi_ref')))
    print(sb[sb.test.isin(['VSE', '5SR'])][['test', 'sex', 'bmi_cat', 'n_pre', 'n_post', 'adj_pct', 'deficit_d', 'deficit_lo', 'deficit_hi', 'p']].round(3).to_string(index=False))

    bc = bmi_change(df); bc.to_csv(os.path.join(SUP, 'S_bmi_change.csv'), index=False)
    print('\n== BMI change ==\n', bc.round(3).to_string(index=False))

    tr = yearly_trend(df); tr.to_csv(os.path.join(SUP, 'S_yearly_trend.csv'), index=False)
    print('\n== Yearly adjusted means ==')
    print(tr.pivot(index='year', columns='test', values='adj_mean').round(2).to_string())
    print(tr.groupby('test').pre_slope_per_year.first().round(3).to_string())

    ep = to_performance_percentiles(empirical_percentiles(df)); ep.to_csv(os.path.join(SUP, 'S_empirical_percentiles.csv'), index=False)
    print('\n== Empirical percentiles (VSE, SLS, 5SR; P10/P50/P90; selected bins) ==')
    sel = ep[(ep.percentile.isin([10, 50, 90])) & (ep.age_bin.isin(['42–47', '54–59', '66–71', '78–84']))]
    print(sel[sel.test.isin(['VSE', 'SLS', '5SR'])][['test', 'age_bin', 'n_pre', 'n_post', 'percentile', 'pre', 'post', 'diff']].round(2).to_string(index=False))

    bp = deficit_columns(to_performance_percentiles(band_percentile_summary(df))); bp.to_csv(os.path.join(TAB, 'table4_band_percentiles.csv'), index=False)
    print('\n== Table 4: percentile differences averaged over 6-month age bands ==')
    print(bp.round(2).to_string(index=False))

    qr = add_deficit(to_performance_percentiles(quantile_regression(df), col='tau', top=1), 'coef', 'lo', 'hi', 'deficit'); qr.to_csv(os.path.join(SUP, 'S_quantile_regression.csv'), index=False)
    print('\n== Quantile regression (complementary analysis; adjusted period difference by tau) ==')
    print(qr.round(3).to_string(index=False))

    pre_rep, rep_counts = pre_repeat_flags(raw0)
    if rep_counts is None:
        print('\n(repeat-measurement file not found: first-measurement analysis removes post-pandemic repeats only)')
    else:
        rep_counts['pre_flagged_in_analytic_sample'] = int(df.index.isin(pre_rep).sum())
        for y, n in df[df.index.isin(pre_rep)].groupby('year').size().items():
            rep_counts[f'pre_flagged_{int(y)}'] = int(n)
        pd.DataFrame([rep_counts]).to_csv(os.path.join(SUP, 'S_pre_repeat_link.csv'), index=False)
        print('\n== Pre-pandemic repeat measurements identified by linkage ==\n', rep_counts)
    ra = repeat_advantage(df, pre_rep); ra.to_csv(os.path.join(SUP, 'S_repeat_advantage.csv'), index=False)
    print('\n== Later vs first measurements (adjusted, SD; + = later better) ==\n', ra.round(3).to_string(index=False))
    rb = robustness(raw, df, pre_rep); rb.to_csv(os.path.join(TAB, 'table5_robustness.csv'))
    print('\n== Robustness (standardised deficit; + = worse) ==\n', rb.round(3).to_string())

    rc = recording_checks(df); rc.to_csv(os.path.join(SUP, 'S_recording_checks.csv'), index=False)
    print('\n== Recording format and standardised difference under common precision / 60-s cap ==\n', rc.round(3).to_string(index=False))

    de = design_effect(df, ov); de.to_csv(os.path.join(SUP, 'S_design_effect.csv'), index=False)
    print(chr(10) + '== Illustrative clustering scenario (not a bound) ==' + chr(10), de.round(3).to_string(index=False))

    ms = missingness(df); ms.to_csv(os.path.join(SUP, 'S_missingness.csv'), index=False)
    print('\n== Missingness vs composite fitness of other tests ==\n', ms.round(3).to_string(index=False))

    rs = repeat_structure(df); rs.to_csv(os.path.join(SUP, 'S_repeat_structure.csv'))
    print('\n== Repeat-measurement index by year ==\n', rs.to_string())

    sbi = sex_bmi_interactions(df); sbi.to_csv(os.path.join(SUP, 'S_sex_bmi_interactions.csv'), index=False)
    print('\n== Period x sex / period x BMI interactions ==\n', sbi.round(4).to_string(index=False))


    lr = low_range_share(df); lr.to_csv(os.path.join(SUP, 'S_low_range_share.csv'), index=False)
    print('\n== Share of records beyond the pre-pandemic 10th/90th percentile (low / high fitness range) ==\n', lr.round(2).to_string(index=False))

    bs = deficit_columns(band_sensitivity(raw)); bs.to_csv(os.path.join(SUP, 'S_percentile_sensitivity.csv'), index=False)
    print('\n== Band-averaged percentile differences under alternative cleaning (VSE, SLS) ==\n', bs.round(2).to_string(index=False))

    s7 = vse_empirical_vs_published(df); s7.to_csv(os.path.join(SUP, 'S_vse_empirical_vs_published.csv'), index=False)
    print('\n== VSE empirical (2017-2019 records) vs published percentiles ==\n', s7.round(1).to_string(index=False))
    # input for gamlss_compare.R (GAMLSS centiles of the same records, Table S7).
    # Record-level data: written to output_v2/private/ for local use only; never distribute.
    priv = os.path.join(OUT, 'private'); os.makedirs(priv, exist_ok=True)
    df[(df.post == 0) & df.endu.notna()][['sex', 'age', 'endu']].to_csv(os.path.join(priv, 'vse_pre_records.csv'), index=False)

    fig_yearly(tr); fig_age(ag); fig_quantile(bp); fig_flow(flow, audit, df)
    print('\nDone. Outputs in', OUT)


if __name__ == '__main__':
    main()
