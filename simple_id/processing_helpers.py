import pandas

socialSSs = {'ANTHROPOLOGY': 'Sociology',
     'AREA STUDIES': 'Social and economic geography',
     'ASIAN STUDIES': 'Other social sciences',
     'BEHAVIORAL SCIENCES': 'Psychology',
     'BUSINESS': 'Economics and business',
     'BUSINESS, FINANCE': 'Economics and business',
     'COMMUNICATION': 'Media and communication',
     'CRIMINOLOGY & PENOLOGY': 'Law',
     'CULTURAL STUDIES': 'Other social sciences',
     'DEMOGRAPHY': 'Sociology',
     'ECONOMICS': 'Economics and business',
     'EDUCATION & EDUCATIONAL RESEARCH': 'Educational sciences',
     'EDUCATION, SCIENTIFIC DISCIPLINES': 'Educational sciences',
     'EDUCATION, SPECIAL': 'Educational sciences',
     'ENVIRONMENTAL STUDIES': 'Social and economic geography',
     'ERGONOMICS': 'Psychology',
     'ETHNIC STUDIES': 'Sociology',
     'FAMILY STUDIES': 'Sociology',
     'GEOGRAPHY': 'Social and economic geography',
     'HOSPITALITY, LEISURE, SPORT & TOURISM': 'Other social sciences',
     'INDUSTRIAL RELATIONS & LABOR': 'Economics and business',
     'INFORMATION SCIENCE & LIBRARY SCIENCE': 'Media and communication',
     'INTERNATIONAL RELATIONS': 'Political science',
     'LAW': 'Law',
     'MANAGEMENT': 'Economics and business',
     'OPERATIONS RESEARCH & MANAGEMENT SCIENCE': 'Economics and business',
     'PLANNING & DEVELOPMENT': 'Social and economic geography',
     'POLITICAL SCIENCE': 'Political science',
     'PSYCHOLOGY': 'Psychology',
     'PSYCHOLOGY, APPLIED': 'Psychology',
     'PSYCHOLOGY, BIOLOGICAL': 'Psychology',
     'PSYCHOLOGY, DEVELOPMENTAL': 'Psychology',
     'PSYCHOLOGY, EDUCATIONAL': 'Psychology',
     'PSYCHOLOGY, EXPERIMENTAL': 'Psychology',
     'PSYCHOLOGY, MATHEMATICAL': 'Psychology',
     'PSYCHOLOGY, MULTIDISCIPLINARY': 'Psychology',
     'PSYCHOLOGY, SOCIAL': 'Psychology',
     'PUBLIC ADMINISTRATION': 'Political science',
     'SOCIAL ISSUES': 'Sociology',
     'SOCIAL SCIENCES, INTERDISCIPLINARY': 'Other social sciences',
     'SOCIAL SCIENCES, MATHEMATICAL METHODS': 'Sociology',
     'SOCIAL WORK': 'Sociology',
     'SOCIOLOGY': 'Sociology',
     'TRANSPORTATION': 'Social and economic geography',
     'URBAN STUDIES': 'Social and economic geography',
     "WOMEN'S STUDIES": 'Sociology'
 }

subjects = sorted(list(set(socialSSs.values())))

fullFname = '../data/full_ss.tsv'
csJournsFname = '../data/CS_journs.tsv'
authsFname = '../data/ss_auth_dat.tsv'
wos_classifications = '../data/wos_classes.csv'

def pprintRow(df, i):
    try:
        r = df.iloc[i]
    except TypeError:
        r = df.loc[i]
    print(r['title'])
    print("{:.2f} {} | {}".format(r['probPos'], r['subject_con'], r['source']))
    print(r['abstract'])

def getSubClass(s):
    if len(s.split('.')) > 1:
        return int(s.split('.')[1][:2])
    else:
        return 0

def read_WOS_CLasses(path):
    df = pandas.read_csv(path)
    df['main_class'] = df['Description'].apply(lambda x: int(x[0]))
    df['sub_class'] = df['Description'].apply(getSubClass)
    df['Description'] = df['Description'].apply(lambda x : ' '.join(x.split(' ')[1:]))
    codeToName = {r['main_class'] if r['sub_class'] < 1 else -1 : r['Description'] for i, r in df.iterrows()}
    df['main_class'] = df['main_class'].apply(lambda x : codeToName[x])
    return df

def get_classes(s):
    try:
        s = s.upper()
    except AttributeError:
        return set()
    vals = set()
    for k in socialSSs.keys():
            if k in s:
                vals.add(socialSSs[k])
    return sorted(list(vals))

def loadSubject(targetSubject, fullFname, csJournsFname, wos_classifications):
    #load full datasets
    df_full = pandas.read_csv(fullFname, sep = '\t', error_bad_lines = False)
    df_cs = pandas.read_csv(csJournsFname, sep = '\t', error_bad_lines = False)
    wos_classes = read_WOS_CLasses(wos_classifications)

    #Label as CS
    CSs = {r['WoS_Description']: r['Description'] for i, r  in wos_classes.iterrows() if r['Description'] == 'Computer and information sciences' and r['sub_class'] > 0}
    socialSSs = {r['WoS_Description']: r['Description'] for i, r  in wos_classes.iterrows() if r['main_class'] == 'SOCIAL SCIENCES' and r['sub_class'] > 0}
    comp_sources = set(df_cs['source'])
    soc_sources = set(df_full['source'])
    df_full['is_comp'] = df_full['source'].apply(lambda x: True if x in comp_sources else False)

    #Get subject labels
    df_ssJourns = df_full[['source', 'subject_con']].groupby('source').max()
    df_ssJourns['subjects'] = df_ssJourns['subject_con'].apply(get_classes)
    journToSubjects = {i : r['subjects'] for i, r in df_ssJourns.iterrows()}
    df_full[targetSubject] = df_full['source'].apply(lambda x: True if targetSubject in journToSubjects[x] else False)
    return df_full[df_full[targetSubject]]


def get_classes(s):
    try:
        s = s.upper()
    except AttributeError:
        return set()
    vals = set()
    for k in socialSSs.keys():
            if k in s:
                vals.add(socialSSs[k])
    return sorted(list(vals))

def getFull():
    df_full = pandas.read_csv(fullFname, sep = '\t', error_bad_lines = False)
    df_cs = pandas.read_csv(csJournsFname, sep = '\t', error_bad_lines = False)
    wos_classes = read_WOS_CLasses(wos_classifications)

    #Label as CS
    CSs = {r['WoS_Description']: r['Description'] for i, r  in wos_classes.iterrows() if r['Description'] == 'Computer and information sciences' and r['sub_class'] > 0}
    comp_sources = set(df_cs['source'])
    soc_sources = set(df_full['source'])
    df_full['is_comp'] = df_full['source'].apply(lambda x: True if x in comp_sources else False)

    #Get subject labels
    df_ssJourns = df_full[['source', 'subject_con']].groupby('source').max()
    df_ssJourns['subjects'] = df_ssJourns['subject_con'].apply(get_classes)
    journToSubjects = {i : r['subjects'] for i, r in df_ssJourns.iterrows()}
    for targetSubject in subjects:
        df_full[targetSubject] = df_full['source'].apply(lambda x: True if targetSubject in journToSubjects[x] else False)
    df_full.index = df_full['wos_id']
    return df_full

def mergePredictions(df, num = 30):
    subDict = {}
    df.index = df['wos_id']
    for subject in subjects:
        try:
            rankings = pandas.read_csv("{}-models-2/rets-{}.csv".format(subject, num), index_col=0)
        except FileNotFoundError:
            continue
        #rankings = rankings[['prediction', 'probPos']]
        #rankings.columns = ['prediction_{}'.format(subject), 'probPos_{}'.format(subject)]
        subDict[subject] = df.join(rankings, how = 'inner', rsuffix='_{}'.format(subject))
        print("Merged {}".format(subject))

    return subDict
