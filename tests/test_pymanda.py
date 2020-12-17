import pytest
from pymanda import ChoiceData, DiscreteChoice
import pandas as pd
import numpy as np


@pytest.fixture
def psa_data():
    '''create data for psa analysis'''
    #create corporations series
    corps = ['x' for x in range(50)]
    corps += ['y' for x in range(25)]
    corps += ['z' for x in range(25)]
    
    #create choice series
    #corp x
    choices = ['a' for x in range(30)]
    choices += ['b' for x in range(20)]
    #corp y
    choices += ['c' for x in range(20)]
    choices += ['d' for x in range(5)]
    #corp z
    choices += ['e' for x in range(25)]
    
    # create zips
    #corp x
    zips = [1 for x in range(20)] #in 75 psa
    zips += [2 for x in range(10)] #in 75 psa
    zips += [3 for x in range(8)] #in 75 psa
    zips += [4 for x in range(7)] #in 90 psa
    zips += [5 for x in range(3)] # out of psa
    zips += [6 for x in range(2)] # out of psa
    #corp y
    zips += [7 for x in range(10)] #in 75 psa
    zips += [8 for x in range(9)] #in 75 psa
    zips += [3 for x in range(4)] #in 90 psa
    zips += [5 for x in range(2)] #out of psa
    #corp z
    zips += [7 for x in range(10)] #in 75 psa
    zips += [10 for x in range(9)] #in 75 psa
    zips += [3 for x in range(4)] #in 90 psa
    zips += [9 for x in range(1)] #out of psa
    zips += ["" for x in range(1)] #out of psa
    
    psa_data = pd.DataFrame({'corporation': corps,
                             'choice' : choices,
                             "geography": zips})
    
    return psa_data

@pytest.fixture()
def onechoice_data():
    choices = ['a' for x in range(100)]
    
    zips = [1 for x in range(20)]
    zips += [2 for x in range(20)]
    zips += [3 for x in range(20)]
    zips += [4 for x in range(20)]
    zips += [5 for x in range(20)]
    
    wght = [1.5 for x in range(20)]
    wght += [1 for x in range(20)]
    wght += [.75 for x in range(20)]
    wght += [.5 for x in range(20)]
    wght += [.25 for x in range(20)]
    
    onechoice_data = pd.DataFrame({'choice': choices,
                                   'geography': zips,
                                   'weight' : wght})
    return onechoice_data
    
## Tests for ChoiceData Initialization
def test_BadInput():
    '''Test for error catching bad input'''
    with pytest.raises(TypeError):
        ChoiceData(["bad", "input"]) 

def test_AllMissing():
    '''test for empty dataframe'''
    df_empty = pd.DataFrame({'corporation': [],
                             'choice' : [],
                             "geography": []})
    with pytest.raises(ValueError):
        ChoiceData(df_empty, 'choice', corp_var='corporation', geog_var='geography')

def test_ChoiceMissing(psa_data):
    '''test for an observation missing choice'''
    df_miss = pd.DataFrame({'corporation': [""],
                             'choice' : [""],
                             "geography": [""]})
    df_miss = pd.concat([df_miss, psa_data])
    with pytest.raises(ValueError):
        ChoiceData(df_miss, 'choice', corp_var='corporation', geog_var='geography')

def test_BadCorp(psa_data):
    '''test for corporation parameter not in data'''
    with pytest.raises(KeyError):
        ChoiceData(psa_data, 'choice', corp_var='corporations', geog_var='geography')

def test_BadGeo(psa_data):
    '''test for geog_var parameter not in data'''
    with pytest.raises(KeyError):
        ChoiceData(psa_data, 'choice', corp_var='corporation', geog_var='zips') 
        
def test_UndefinedCorp(psa_data):
    '''test for empty corporation parameter returning as choice_var'''
    data = ChoiceData(psa_data, 'choice', geog_var='geography')
    assert data.corp_var== "choice"
    

## Tests for estimate_psas()
@pytest.fixture
def cd_psa(psa_data):
    cd_psa = ChoiceData(psa_data, "choice", corp_var='corporation', geog_var='geography')
    return cd_psa

def test_define_geogvar(psa_data):
    '''Test for error raising if geography variable is not defined'''
    bad_cd = ChoiceData(psa_data, "choice", corp_var='corporation')
    with pytest.raises(KeyError):
        bad_cd.estimate_psa(["x"])

def test_BadThreshold(cd_psa):
    '''Test for error raising if thresholds are not between 0 and 1'''
    with pytest.raises(TypeError):
        cd_psa.estimate_psa(['x'], threshold=[.75, .9, 75])

def test_BadCenters(cd_psa):
    '''Test for if psa centers are not in corp_var'''
    with pytest.raises(ValueError):
        cd_psa.estimate_psa(['a'])

def test_3Corp(cd_psa):
    '''Estimate the 3 Corporations in the psa data with default parameters'''
    psa_dict = cd_psa.estimate_psa(['x', 'y', 'z'])
    
    answer_dict = {'x_0.75': [1,2,3],
                   'x_0.9': [1,2,3,4],
                   'y_0.75': [7,8],
                   'y_0.9': [3,7,8],
                   'z_0.75': [7,10],
                   'z_0.9' : [3,7,10]}
    
    assert answer_dict==psa_dict

def test_1corp(cd_psa):
    '''Estimate the 1 Corporation in the psa data with default parameters'''
    psa_dict = cd_psa.estimate_psa(['x'])
    
    answer_dict = {'x_0.75': [1,2,3],
                   'x_0.9': [1,2,3,4]}
    
    assert answer_dict==psa_dict
    
def test_1corp_weight(onechoice_data):
    ''' Estimate PSAs for 1 corporation with weight var and custom threshold'''
    cd_onechoice = ChoiceData(onechoice_data, 'choice', geog_var='geography', wght_var='weight')
    
    psa_dict = cd_onechoice.estimate_psa(['a'], threshold=.6)
    
    answer_dict = {'a_0.6': [1,2]}
    
    assert psa_dict==answer_dict
    
def test_MultipleThresholds(onechoice_data):
    ''' Estimate PSAs for multiple custom thresholds'''
    cd_onechoice = ChoiceData(onechoice_data, 'choice', geog_var='geography', wght_var='weight')
    
    psa_dict = cd_onechoice.estimate_psa(['a'], threshold=[.6, .7])
    
    answer_dict = {'a_0.6': [1,2],
                   'a_0.7': [1,2,3]}
    
    assert psa_dict == answer_dict

##Tests for restrict_data
def test_RestrictData(psa_data, cd_psa):
    '''Check if restrict data restricts data properly'''
    cd_psa.restrict_data(psa_data['corporation']=='x')
    
    restricted_data = psa_data[psa_data['corporation']=='x']
    cd_restricted = ChoiceData(restricted_data, "choice", corp_var='corporation', geog_var='geography')
    assert cd_restricted.data.equals(restricted_data)
    
def test_BadSeries(cd_psa, psa_data):
    '''Restrict_data should only accept boolean series'''
    flag_series = np.where(psa_data['corporation']=='x', 1, 0)
    with pytest.raises(TypeError):
        cd_psa.restrict_data(flag_series)

# Tests for calculate shares
def dictionary_comparison(dict1, dict2):
    if len(dict1.keys()) == 0 or len(dict2.keys()) == 0:
        raise ValueError("Comparisons have 0 Keys")
    
    matches = []
    if dict1.keys() == dict2.keys():
        for key in dict1.keys():
            matches.append(dict1[key].equals(dict2[key])) 
    else:
        raise ValueError("Comparison Keys do not match")
    
    return all(matches)

def test_BaseShares(cd_psa, base_shares):
    test_shares = cd_psa.calculate_shares()
    actual_shares = {'Base Shares': pd.DataFrame(base_shares)}
    
    assert dictionary_comparison(test_shares, actual_shares)
    
def test_PsaShares(cd_psa, psa_shares):
    psa_test = {'x_0.75': [1,2,3]}
    test_shares = cd_psa.calculate_shares(psa_test)
    
    actual_shares = {'x_0.75': psa_shares}
    
    assert dictionary_comparison(test_shares, actual_shares)
    
def test_MultiplePsaShares(cd_psa, MultiplePsaShares):
    psa_test = {'x_0.75': [1,2,3],
                'x_0.9': [1,2,3,4]}
    
    test_shares = cd_psa.calculate_shares(psa_test)
    
    actual_shares = MultiplePsaShares
    assert dictionary_comparison(test_shares, actual_shares)

#Tests for exporting Shares
def test_ExportBaseShares(cd_psa, base_shares):
    base_dict = {"Base Shares": base_shares}
    test = cd_psa.export_shares(base_dict, export=False)
    
    base = [100, 50, 30, 20, 25, 20, 5, 25, 24]
    
    answer_df = pd.DataFrame({"corporation": ["Total", "x", "x", "x", "y", "y", "y", "z", "z"],
                           "choice": ['Total', 'Total', 'a', 'b', 'Total', 'c', 'd', 'Total', 'e'],
                           "count_Base Shares": base,
                           "count_share_Base Shares": [x / 100 for x in base]})
    answer = {'Base Shares': answer_df}    
                                       
    return dictionary_comparison(test, answer)

def test_ExportMultipleThresholds(cd_psa, MultiplePsaShares, MultiplePsaExport):
    test = cd_psa.export_shares(MultiplePsaShares, export=False)
    
    answer = MultiplePsaExport
    return dictionary_comparison(test, answer)

def test_ExportMultiplePSAs(cd_psa, MultiplePsaShares, MultiplePsaExport):
    psas = MultiplePsaShares
    
    z75 = [10,19]
    z90 = [8, 11, 3, 23]
    
    psas.update({ 'z_0.75': pd.DataFrame({'corporation': ['y', 'z'],
                                         'choice': ['c', 'e'],
                                         'count': z75,
                                         'count_shares': [x / 29 for x in z75]}),
        'z_0.9': pd.DataFrame({'corporation': ['x', 'y', 'y', 'z'],
                                         'choice': ['b', 'c', 'd', 'e'],
                                         'count': z90,
                                         'count_shares': [x / 56 for x in z90]})})
    
    test = cd_psa.export_shares(psas, export=False)
    
    answer = MultiplePsaExport
    z75 = [29, 19, 19, 10, 10, 0, 0, 0]
    z90 = [45, 23, 23, 14, 11, 3, 8, 8]
    z_df = pd.DataFrame({'corporation': ['Total', 'z', 'z', 'y', 'y', 'y', 'x', 'x'],
                                    'choice': ['Total', 'Total', 'e', 'Total', 'c', 'd', 'Total', 'b'],
                                    'count_0.75': z75,
                                    'count_share_0.75': [x / 29 for x in z75],
                                    'count_0.9': z90,
                                    'count_share_0.9': [x / 45 for x in z90]})
    answer.update({"z": z_df})
    
    return dictionary_comparison(test, answer)
    


#test for calculating HHI shares
@pytest.fixture
def base_shares():
    base_shares = pd.DataFrame({'corporation':['x', 'x', 'y', 'y', 'z'],
                                'choice': ['a', 'b', 'c', 'd', 'e'],
                                'count': [30, 20, 20, 5, 25],
                                'count_share': [.3, .2, .2, .05, .25]})
    return base_shares

@pytest.fixture
def base_hhi():
    base_hhi = 3750.0
    return base_hhi
    
@pytest.fixture
def psa_shares():
    psa_shares = pd.DataFrame({'corporation': ['x', 'x', 'y', 'y', 'z'],
                                'choice': ['a', 'b', 'c', 'd', 'e'],
                                'count': [x for x in [30, 8, 1,3, 4]],
                                'count_share': [x / 46 for x in [30, 8, 1,3, 4]]})
    return psa_shares

@pytest.fixture
def MultiplePsaShares(psa_shares):
    MultiplePsaShares = {'x_0.75': psa_shares,
                'x_0.9': pd.DataFrame({'corporation': ['x', 'x', 'y', 'y', 'z'],
                                        'choice': ['a', 'b', 'c', 'd', 'e'],
                                        'count': [x for x in [30, 15, 1,3,4]],
                                        'count_share': [x / 53 for x in [30, 15, 1,3,4]]
                                        })}
    return MultiplePsaShares

@pytest.fixture
def MultiplePsaExport():
    x_75 = [46, 38, 30, 8, 4, 3, 1, 4, 4]
    x_90 = [53, 45, 30, 15, 4, 3, 1, 4, 4]
    MultiplePsaExport = {'x': pd.DataFrame({'corporation': ['Total', 'x', 'x', 'x', 'y', 'y', 'y', 'z', 'z'],
                          'choice': ['Total', 'Total', 'a', 'b', 'Total', 'd', 'c', 'Total', 'e'],
                          'count_0.75': x_75,
                          'count_share_0.75': [x / 46 for x in x_75],
                          'count_0.9': x_90,
                          'count_share_0.9': [x / 53 for x in x_90]})}
    return MultiplePsaExport

@pytest.fixture
def psa_hhi():
    psa_hhi = 479347578744499 / 68719476736 # approximately 6975.43
    return psa_hhi

def test_HHIs(cd_psa, base_shares, psa_shares, base_hhi, psa_hhi):
    share_tables = {'Base Shares': base_shares,
                    'x_0.75': psa_shares}
    
    test_hhis = cd_psa.calculate_hhi(share_tables)
    
    actual_hhis = {'Base Shares': base_hhi,
                  'x_0.75': psa_hhi} 
    
    assert test_hhis == actual_hhis
    
def test_HHI_sharecol(cd_psa, base_shares, psa_shares, base_hhi):
    df_alt = base_shares
    df_alt['other shares'] = df_alt['count_share']
    df_alt = df_alt.drop(columns="count_share")
    
    share_tables = {'Base Shares': df_alt}
    
    test_hhis = cd_psa.calculate_hhi(share_tables, share_col="other shares")
    
    actual_hhis = {'Base Shares': base_hhi}
    
    assert test_hhis == actual_hhis
    
def test_HHI_BadSharecol(cd_psa, base_shares, psa_shares):
    df_alt = psa_shares
    df_alt['other shares'] = df_alt['count_share']
    df_alt = df_alt.drop(columns="count_share")
    
    share_tables = {'Base Shares': base_shares,
                'x_0.75': df_alt}
    
    with pytest.raises(KeyError):
        cd_psa.calculate_hhi(share_tables)

def test_HHI_BadInput(cd_psa):
    with pytest.raises(TypeError):
        cd_psa.calculate_hhi(cd_psa.data)

# test for calculating changes in HHI
@pytest.fixture
def hhi_index():
    hhi_index = ['Pre-Merger HHI', 'Post-Merger HHI', 'HHI Change']
    
    return hhi_index

@pytest.fixture
def MultiHHIChange(base_hhi, psa_hhi, hhi_index):
    MultiHHIChange_df = pd.DataFrame({'Base Shares': [base_hhi, 5000, 1250],
                 'x_0.75' : [psa_hhi, 7835839010804385 / 1099511627776, 166277750892401 / 1099511627776]}, # approximately 7126.66 post and 151.23 change
                 index=hhi_index)
    
    MultiHHIChange = {"transaction": MultiHHIChange_df}
    return MultiHHIChange

def test_HHIChange(cd_psa, base_shares, psa_shares, MultiHHIChange):
    share_dict = {'Base Shares': base_shares,
                  'x_0.75': psa_shares}
    test_change = cd_psa.hhi_change({"transaction": ['y', 'z']}, share_dict)
    
    actual_change = MultiHHIChange
    assert dictionary_comparison(test_change, actual_change)

def test_HHIChange_TransCol(cd_psa, base_shares, base_hhi, hhi_index):

    share_dict = {"Base Shares": base_shares}
    
    test_change = cd_psa.hhi_change({'transaction': ['d', 'e']}, share_dict, trans_var="choice")
    
    actual_change_df = pd.DataFrame({'Base Shares': [base_hhi, 3800, 50]},
                                 index=hhi_index)
    
    actual_change = {'transaction': actual_change_df}
    
    assert dictionary_comparison(test_change, actual_change)
    
def test_HHIChange_MultipleTrans(cd_psa, base_shares, base_hhi, hhi_index):
    share_dict = {"Base Shares": base_shares}
    
    transactions = {"trans1": ['d', 'e'],
                    "trans2": ["c", "d", "e"]}
    
    test_change = cd_psa.hhi_change(transactions, share_dict, trans_var="choice")
    
    actual_change = {'trans1': pd.DataFrame({"Base Shares": [base_hhi, 3800, 50]}, index=hhi_index),
                     'trans2': pd.DataFrame({"Base Shares": [base_hhi, 5000, 1250]}, index=hhi_index)}
    
    assert dictionary_comparison(test_change, actual_change)

def test_HHIChange_BadList(cd_psa, base_shares):
    share_dict = {"Base Shares": base_shares}
    
    with pytest.raises(ValueError):
        cd_psa.hhi_change({"trans1":['c', 'd', 'e']}, share_dict)

def test_HHIChange_NotList(cd_psa, base_shares):
    share_dict = {"Base Shares": base_shares}

    with pytest.raises(TypeError):
        cd_psa.hhi_change({'trans1':'x'}, share_dict)
    
def test_HHIChange_BadTransCol(cd_psa, base_shares):
    share_dict = {"Base Shares": base_shares}

    with pytest.raises(KeyError):
        cd_psa.hhi_change({'trans1':['c', 'd']}, share_dict, trans_var='systen')

# Tests for export HHI
def test_ExportHHI_Change(cd_psa, MultiHHIChange, hhi_index):
    test = cd_psa.export_hhi_change(MultiHHIChange, export=False)
    changes= MultiHHIChange['transaction']
    
    answer = {'transaction': pd.DataFrame({"Base Shares": changes['Base Shares'],
                                           "x_0.75": changes['x_0.75']}, index=hhi_index)}
    
    assert dictionary_comparison(test, answer)
    
# Tests for stratify_shares
@pytest.fixture
def stratified_share_counts(): # based on semi_cd_corp
    stratified_share_counts = pd.DataFrame({"corp": ["Total", "a", 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
                                            "choice": ['Total', 'Total', 'u', 'v', 'Total', 'w', 'x', 'Total', 'y', 'z'],
                                            "Row Totals":[1000, 560, 300, 260, 220, 110, 110, 220, 110, 110],
                                            "0_counts":[500, 220, 220, 0, 140, 110, 30, 140, 110, 30],
                                            "1_counts":[500, 340, 80, 260, 80, 0, 80, 80, 0, 80]},
                                             dtype="object")
    
    stratified_share_counts = {'Base Stratified': stratified_share_counts}
    return stratified_share_counts

@pytest.fixture
def stratified_share_cols(): # based on semi_cd_corp
    stratified_share_cols = pd.DataFrame({"0_0":[x / 500 for x in [220, 0, 110, 30, 110, 30]],
                                          "1_0":[x / 500 for x in[80, 260, 0, 80, 0, 80]]},
                                     index = ['u', 'v', 'w', 'x', 'y', 'z'])
    
    stratified_share_cols.index.name = "choice"
    stratified_share_cols['corp'] = ['a', 'a', 'b', 'b', 'c', 'c']
    stratified_share_cols = stratified_share_cols.reset_index()
    stratified_share_cols = stratified_share_cols.set_index(['corp', 'choice'])
    stratified_share_cols =stratified_share_cols.reset_index()
    
    stratified_share_cols = {'Base Stratified': stratified_share_cols}
    
    return stratified_share_cols

@pytest.fixture
def stratified_share_rows(stratified_share_counts): # based on semi_cd_corp
    df = stratified_share_counts['Base Stratified']
    df = df[~df['choice'].isin(["Total"])]
    df = df.set_index(["corp", "choice"])
    df = df.drop(columns=["Row Totals"])
    stratified_share_rows = (df.T / df.sum(axis=1)).T
    stratified_share_rows= stratified_share_rows.rename(columns = {"0_counts": "0_1",
                                                         "1_counts" : "1_1"})
    stratified_share_rows = stratified_share_rows.reset_index()
    stratified_share_rows['0_1'] = stratified_share_rows['0_1'].astype('float')
    stratified_share_rows['1_1'] = stratified_share_rows['1_1'].astype('float')

    stratified_share_rows = {'Base Stratified': stratified_share_rows}
    return stratified_share_rows

@pytest.fixture
def startified_share_count_cols():
    startified_share_count_cols = pd.DataFrame({"0_counts":[220, 0, 110, 30, 110, 30],
                                                "0_0":[x / 500 for x in [220, 0, 110, 30, 110, 30]],
                                                "1_counts":[80, 260, 0, 80, 0, 80],
                                                "1_0":[x / 500 for x in[80, 260, 0, 80, 0, 80]]},
                                           index = ['u', 'v', 'w', 'x', 'y', 'z'])
    
    startified_share_count_cols.index.name = "choice"
    startified_share_count_cols['corp'] = ['a', 'a', 'b', 'b', 'c', 'c']
    startified_share_count_cols = startified_share_count_cols.reset_index()
    startified_share_count_cols = startified_share_count_cols.set_index(['corp', 'choice'])    
    
    startified_share_count_cols = {'Base Stratified': startified_share_count_cols}
    
    return startified_share_count_cols

@pytest.fixture
def strat_cd(psa_data):
    strat_psa = psa_data
    
    stratified = [1 for x in range(15)]
    stratified += [2 for x in range(10)]
    stratified += [3 for x in range(13)]
    stratified += [4 for x in range(12)] # out of psa x_75
    
    stratified += [2 for x in range(19)] # out
    stratified += [1 for x in range(2)]
    stratified += [2 for x in range(2)] 
    stratified += [3 for x in range(2)] # out
    
    stratified += [4 for x in range(19)] # out
    stratified += [3 for x in range(4)]
    stratified += [2 for x in range(2)] # out
    
    strat_psa['stratified'] = stratified
    
    strat_cd = ChoiceData(strat_psa, 'choice', 'corporation', 'geography')
    
    return strat_cd
    

def test_StratifiedShares(stratified_share_counts, semi_cd_corp):
    test = semi_cd_corp.stratify_shares("x1")

    answer = stratified_share_counts
    answer['Base Stratified'] = answer['Base Stratified'].drop(columns=["Row Totals"])
    
    assert dictionary_comparison(test, answer)
    
def test_StratifiedShares_RowsAsLevels(stratified_share_counts, semi_cd_corp):
    test = semi_cd_corp.stratify_shares("x1", rows_as_levels=True, subtotals=False, col_totals=False)

    answer = pd.DataFrame({0:[220, 0, 110, 30, 110, 30],
                           1:[80, 260, 0, 80, 0, 80]},
                            index = ['u_counts', 'v_counts', 'w_counts', 'x_counts', 'y_counts', 'z_counts'])
 
    answer.index.name = "choice"
    answer['corp'] = ['a', 'a', 'b', 'b', 'c', 'c']
    answer = answer.reset_index()
    answer = answer.set_index(['corp', 'choice'])  
    answer = answer.T
    answer.index.name = "x1"
    answer = answer.reset_index()
    answer = {'Base Stratified': answer}
    
    assert dictionary_comparison(test, answer)
    
def test_StratifiedShares_Axis0(stratified_share_cols, semi_cd_corp):
    test = semi_cd_corp.stratify_shares("x1", shares_axis=0, subtotals=False, col_totals=False)

    answer = stratified_share_cols
    
    assert dictionary_comparison(test, answer)
        
def test_StratifiedShares_Axis1(stratified_share_rows, semi_cd_corp):
    test = semi_cd_corp.stratify_shares("x1", shares_axis=1, subtotals=False, col_totals=False)

    answer = stratified_share_rows
    
    assert dictionary_comparison(test, answer)
    
def test_StratifiedShares_RowsAsLevels_Axis0(semi_cd_corp, stratified_share_cols):
    test = semi_cd_corp.stratify_shares("x1", shares_axis=0, subtotals=False, col_totals=False)
    
    answer = stratified_share_cols

    assert dictionary_comparison(test, answer)

def test_StratifiedShares_RowsAsLevels_Axis1(semi_cd_corp, stratified_share_rows):
    test = semi_cd_corp.stratify_shares("x1", shares_axis=1, subtotals=False, col_totals=False)
    
    answer = stratified_share_rows

    assert dictionary_comparison(test, answer)
    
def test_StratifiedShares_multiaxis(semi_cd_corp, startified_share_count_cols):
    test = semi_cd_corp.stratify_shares("x1", shares_axis=['counts', 0], subtotals=False, col_totals=False)
    
    answer = startified_share_count_cols
    answer['Base Stratified'] = answer['Base Stratified'].reset_index()

    assert dictionary_comparison(test, answer)
    
    
# Tests for export stratified
def test_Stratified_MultipleAxesError(semi_cd_corp, startified_share_count_cols):
    
    with pytest.raises(ValueError):
        semi_cd_corp.stratify_shares(levels_var="x1", shares_axis=["counts", 1], row_totals=True)
        

def test_Stratified_columntotals(semi_cd_corp, startified_share_count_cols):
    
    test = semi_cd_corp.stratify_shares("x1", shares_axis=['counts', 0], col_totals=True)

    answer = pd.DataFrame({"0_counts":[500, 220, 220, 0, 140,  110, 30, 140, 110, 30],
                         "0_0":[x / 500 for x in [500, 220, 220, 0, 140,  110, 30, 140, 110, 30]],
                         "1_counts":[500, 340, 80, 260, 80,  0, 80, 80, 0, 80],
                         "1_0":[x / 500 for x in [500, 340, 80, 260, 80,  0, 80, 80, 0, 80]]},
                        index = ['Total', 'Total', 'u', 'v', 'Total', 'w', 'x', 'Total', 'y', 'z'],
                        dtype="object")
    
    answer.index.name = "choice"
    answer['corp'] = ['Total', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
    answer = answer.reset_index()
    answer = answer[['corp', 'choice', '0_counts', '0_0', '1_counts', '1_0']]
    
    answer = {'Base Stratified': answer}
    
    assert dictionary_comparison(test, answer)
    
def test_Stratified_rowtotals(semi_cd_corp, stratified_share_counts):
    
    test = semi_cd_corp.stratify_shares("x1", row_totals=True)
    
    answer = pd.DataFrame({"Row Totals":[1000, 560, 300, 260, 220, 110, 110, 220, 110, 110],
                            "0_counts":[500, 220, 220, 0, 140, 110, 30, 140, 110, 30],
                            "1_counts":[500, 340, 80, 260, 80, 0, 80, 80, 0, 80]},
                             index = ['Total', 'Total', 'u', 'v', 'Total', 'w', 'x', 'Total', 'y', 'z'],
                             dtype="object")
    
    answer.index.name = "choice"
    answer['corp'] = ['Total', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
    answer = answer.reset_index()
    answer = answer[['corp', 'choice', 'Row Totals', '0_counts', '1_counts']]  
    answer['Row Totals'] = answer['Row Totals'].astype("float")
    
    answer = {'Base Stratified': answer}
    
    assert dictionary_comparison(test, answer)

def test_Stratified_outmigrations(strat_cd):
    psa_zips = {'x_0.75': [1,2,3],
                   'x_0.9': [1,2,3,4]}
    
    test = strat_cd.stratify_shares("stratified", rows_as_levels=True, outmigration=True, 
                                    corp_var=True, subtotals=False, psas=psa_zips)
    
    x75 = pd.DataFrame({'stratified': ['Total', 1, 2, 3],
                        'PSA Total': [46, 17, 12, 17],
                        'x_counts': [38, 15, 10, 13],
                        'Outmigration': [8,2,2,4],
                        'Outmigration_Share': [8/46, 2/17, 2/12, 4/17],
                        'y_counts': [4,2,2,0],
                        'z_counts': [4,0,0,4]})
    
    x90 = pd.DataFrame({'stratified': ['Total', 1, 2, 3, 4],
                        'PSA Total': [53, 17, 12, 17, 7],
                        'x_counts': [45, 15, 10, 13, 7],
                        'Outmigration': [8,2,2,4, 0],
                        'Outmigration_Share': [8/53, 2/17, 2/12, 4/17, 0],
                        'y_counts': [4,2,2,0, 0],
                        'z_counts': [4,0,0,4, 0]})
    
    answer = {'x_0.75' : x75,
              'x_0.9': x90}
    
    assert dictionary_comparison(test, answer)

# Tests for DiscreteChoice
@pytest.fixture
def semi_cd():
    choice = ['a' for x in range(560)]
    choice += ['b' for z in range(220)]   
    choice += ['c' for x in range(220)]
    
    #choice a
    x1 = [0 for x in range(220)]
    x1 += [1 for x in range(340)]
    
    x2 = [0 for x in range(200)]
    x2 += [1 for x in range(20)]
    x2 += [0 for x in range(120)]
    x2 += [1 for x in range(220)]
    

    x3 = [2 for x in range(200)]
    x3 += [0 for x in range(120)]
    x3 += [2 for x in range(20)]
    x3 += [0 for x in range(100)]
    x3 += [1 for x in range(100)]
    x3 += [2 for x in range(20)]
    
    #choice b
    x1 += [0 for x in range(140)]
    x1 += [1 for x in range(80)]

    
    x2 += [0 for x in range(60)]
    x2 += [1 for x in range(80)]
    x2 += [0 for x in range(40)]
    x2 += [1 for x in range(40)]
    
    x3 += [0 for x in range(20)]
    x3 += [2 for x in range(40)]
    x3 += [1 for x in range(40)]
    x3 += [2 for x in range(40)]
    x3 += [0 for x in range(40)]
    x3 += [1 for x in range(40)]
    
    # choice c
    x1 += [0 for x in range(140)]
    x1 += [1 for x in range(80)]

    x2 += [0 for x in range(120)]
    x2 += [1 for x in range(100)]

    x3 += [1 for x in range(20)]
    x3 += [2 for x in range(100)]
    x3 += [1 for x in range(20)]
    x3 += [0 for x in range(40)]
    x3 += [1 for x in range(40)]
    
    semi_df = pd.DataFrame({'choice': choice,
                            'x1': x1,
                            'x3': x3,
                            'x2': x2})
    semi_cd = ChoiceData(semi_df, 'choice')

    return semi_cd

@pytest.fixture
def semi_cd_wght():
    choice = ['a' for x in range(56)] # scaled by .1
    choice += ['b' for z in range(110)] # scaled by .5
    choice += ['c' for x in range(440)] # scaled by 2
    
    #choice a
    x1 = [0 for x in range(22)]
    x1 += [1 for x in range(34)]
    
    x2 = [0 for x in range(20)]
    x2 += [1 for x in range(2)]
    x2 += [0 for x in range(12)]
    x2 += [1 for x in range(22)]
    

    x3 = [2 for x in range(20)]
    x3 += [0 for x in range(12)]
    x3 += [2 for x in range(2)]
    x3 += [0 for x in range(10)]
    x3 += [1 for x in range(10)]
    x3 += [2 for x in range(2)]
    
    wght= [10 for x in range(56)]
    
    #choice b
    x1 += [0 for x in range(70)]
    x1 += [1 for x in range(40)]

    
    x2 += [0 for x in range(30)]
    x2 += [1 for x in range(40)]
    x2 += [0 for x in range(20)]
    x2 += [1 for x in range(20)]
    
    x3 += [0 for x in range(10)]
    x3 += [2 for x in range(20)]
    x3 += [1 for x in range(20)]
    x3 += [2 for x in range(20)]
    x3 += [0 for x in range(20)]
    x3 += [1 for x in range(20)]
    
    wght += [2 for x in range(110)]
    
    # choice c
    x1 += [0 for x in range(280)]
    x1 += [1 for x in range(160)]

    x2 += [0 for x in range(240)]
    x2 += [1 for x in range(200)]

    x3 += [1 for x in range(40)]
    x3 += [2 for x in range(200)]
    x3 += [1 for x in range(40)]
    x3 += [0 for x in range(80)]
    x3 += [1 for x in range(80)]
    
    wght += [.5 for x in range(440)]
    
    semi_df_weight = pd.DataFrame({'choice': choice,
                            'x1': x1,
                            'x3': x3,
                            'x2': x2,
                            'weight': wght})
    semi_cd_wght = ChoiceData(semi_df_weight, 'choice', wght_var='weight')

    return semi_cd_wght

@pytest.fixture
def semi_cd_corp(semi_cd):
    df= semi_cd.data.copy()
    df['corp'] = df['choice']
    choices = ['u' for x in range(300)] + ['v' for x in range(260)]
    choices += ['w' for x in range(110)] + ['x' for x in range(110)]
    choices += ['y' for x in range(110)] + ['z' for x in range(110)]
    
    df['choice'] = choices
    
    semi_cd_corp = ChoiceData(df, 'choice', corp_var='corp')
    
    return semi_cd_corp

@pytest.fixture
def semi_dc():
    semi_dc = DiscreteChoice(solver='semiparametric', coef_order = ['x1', 'x2', 'x3'])
    return semi_dc

@pytest.fixture
def semi_div(semi_cd):
    semi_div = pd.DataFrame({'group': ['0\b0','0\b0\b2','0\b1\b1', '0\b1\b2', '1', '1\b0\b0', '1\b1\b0', '1\b1\b1', 'ungrouped'],
                'a': [0, .5882, 0, 0, 1, .7143, .7143, .5556, 1],
                'b': [.5, .1176, .6667, 1, 0, .2857, 0, .2222, 0],
                'c': [.5, .2941, .3333, 0, 0, 0, .2857, .2222, 0]})

    return semi_div
    
def test_DC_semiparam_fit(semi_dc, semi_cd, semi_div):
    semi_dc.fit(semi_cd)
    test = semi_dc.coef_.round(decimals=4)
    
    assert test.equals(semi_div)

def test_DC_semiparam_fit_weight(semi_dc, semi_cd_wght, semi_div):
    semi_dc.fit(semi_cd_wght)
    test = semi_dc.coef_.round(decimals=4)
    
    assert test.equals(semi_div)

def test_DC_semiparam_fit_minbin(semi_cd):
    dc = DiscreteChoice(solver='semiparametric', coef_order = ['x1', 'x2', 'x3'], min_bin=50)
    dc.fit(semi_cd)
    
    test = dc.coef_.round(decimals=4)
    
    actual = pd.DataFrame({'group': ['0\b0\b2','0\b1','0\b1\b1', '1\b0\b0', '1\b1\b0', '1\b1\b1', 'ungrouped'],
                'a': [.5882, .3333, 0, .7143, .7143, .5556, .5],
                'b': [.1176, .6667, .6667, .2857, 0, .2222, .25],
                'c': [.2941, 0, .3333, 0, .2857, .2222, .25]}) 
    
    assert test.equals(actual)
    
#tests for predict()    
def test_DC_semiparam_predict(semi_dc, semi_cd, semi_div):
    
    semi_dc.fit(semi_cd)
    test = semi_dc.predict(semi_cd)

    actual = semi_cd.data.copy()
    actual['group'] = actual[['x1', 'x2', 'x3']].astype(str).agg('\b'.join,axis=1)
    actual = actual.drop(columns=['x1', 'x2','x3', 'choice'])
    actual['group'] = actual['group'].str.replace('0\b1\b0', 'ungrouped')
    actual['group'] = actual['group'].str.replace('0\b0\b0', '0\b0')
    actual['group'] = actual['group'].str.replace('0\b0\b1', '0\b0')
    actual['group'] = actual['group'].str.replace('1\b0\b2', '1')
    actual['group'] = actual['group'].str.replace('1\b1\b2', '1')

    actual = actual.merge(semi_div, how='left', on='group')
    actual = actual.drop(columns=['group'])
    
    assert test.round(decimals=4).equals(actual)
    
def test_DC_semiparam_predict_wght(semi_dc, semi_cd_wght, semi_div):
    
    semi_dc.fit(semi_cd_wght)
    test = semi_dc.predict(semi_cd_wght)

    actual = semi_cd_wght.data.copy()
    actual['group'] = actual[['x1', 'x2', 'x3']].astype(str).agg('\b'.join,axis=1)
    actual = actual.drop(columns=['x1', 'x2','x3', 'choice'])
    actual['group'] = actual['group'].str.replace('0\b1\b0', 'ungrouped')
    actual['group'] = actual['group'].str.replace('0\b0\b0', '0\b0')
    actual['group'] = actual['group'].str.replace('0\b0\b1', '0\b0')
    actual['group'] = actual['group'].str.replace('1\b0\b2', '1')
    actual['group'] = actual['group'].str.replace('1\b1\b2', '1')

    actual = actual.merge(semi_div, how='left', on='group')
    actual = actual.drop(columns=['group', 'weight'])

    assert test.round(decimals=4).equals(actual)

#tests for diversions    

@pytest.fixture
def dc_semiparam_diversion():
    dc_semiparam_diversion = pd.DataFrame({'a': [np.NaN, .4143, .5857],
                           'b': [.5291, np.NaN, .4709],
                           'c': [.6905, .3095, np.NaN]},
                          index = ['a', 'b', 'c'])
    
    return dc_semiparam_diversion
def test_DC_semiparm_diversion(semi_dc, semi_cd, dc_semiparam_diversion):
    
    semi_dc.fit(semi_cd)
    
    choice_probs = semi_dc.predict(semi_cd)  
    test = semi_dc.diversion(semi_cd, choice_probs, div_choices=['a', 'b', 'c'])
    
    actual = dc_semiparam_diversion
    
    assert test.round(decimals=4).equals(actual)
    
def test_DC_semiparm_diversion_wght(semi_dc, semi_cd_wght, dc_semiparam_diversion):
    
    semi_dc.fit(semi_cd_wght)
    
    choice_probs = semi_dc.predict(semi_cd_wght)  
    test = semi_dc.diversion(semi_cd_wght, choice_probs, div_choices=['a', 'b', 'c'])
    
    actual = dc_semiparam_diversion
    
    assert test.round(decimals=4).equals(actual)
    
def test_DC_semiparm_diversion_2choice(semi_dc, semi_cd,dc_semiparam_diversion):
    
    semi_dc.fit(semi_cd)
    
    choice_probs = semi_dc.predict(semi_cd)  
    test = semi_dc.diversion(semi_cd, choice_probs, div_choices=['a', 'b'])
    
    actual = dc_semiparam_diversion.drop(columns=['c'])
    
    assert test.round(decimals=4).equals(actual)

@pytest.fixture
def dc_semiparam_corp_divs():
    dc_semiparam_corp_divs = pd.DataFrame({'a': [np.NaN, np.NaN, .1143, .3, .2571, .3286],
                           'b': [.3259, .2032, np.NaN, np.NaN, .1778, .2931],
                           'c': [.3788, .3117, .2576, .0519, np.NaN, np.NaN]},
                          index = ['u', 'v', 'w', 'x', 'y', 'z'])
    
    return dc_semiparam_corp_divs
    
def test_DC_semiparam_fit_corp(semi_dc, semi_cd_corp, dc_semiparam_corp_divs):
    
    semi_dc.fit(semi_cd_corp)
    choice_probs = semi_dc.predict(semi_cd_corp)
    
    test = semi_dc.diversion(semi_cd_corp, choice_probs, div_choices=['a', 'b', 'c'])
    
    actual = dc_semiparam_corp_divs
    
    assert test.round(decimals=4).equals(actual)
    
@pytest.fixture
def dc_semiparam_corp_divs_choice():
    dc_semiparam_corp_divs_choice = pd.DataFrame({'u': [np.NaN, .0952, .2041, .1905, .4592, .051]},
                          index = ['u', 'v', 'w', 'x', 'y', 'z'])
    
    return dc_semiparam_corp_divs_choice

def test_DC_semiparam_fit_div_shares_col(semi_dc, semi_cd_corp, dc_semiparam_corp_divs_choice):
    
    semi_dc.fit(semi_cd_corp)
    choice_probs = semi_dc.predict(semi_cd_corp)
    
    test = semi_dc.diversion(semi_cd_corp, choice_probs, div_choices=['u'], div_choices_var = semi_cd_corp.choice_var)
    
    actual = dc_semiparam_corp_divs_choice
    
    assert test.round(decimals=4).equals(actual)  
    
# tests for export_diversions
def test_ExportDiversions_corp_to_choice(semi_dc, semi_cd_corp, dc_semiparam_corp_divs):
    semi_dc.fit(semi_cd_corp)

    test = semi_dc.export_diversions(dc_semiparam_corp_divs, semi_cd_corp, export=False)
    
    a = [1, .5857, .3286, .2571, .4143, .3, .1143]
    base = [1000, 220, 110, 110, 220, 110, 110]
    a_df = pd.DataFrame({'corp': ['Total', 'c', 'c', 'c', 'b', 'b', 'b'],
                         'choice': ['Total', 'Total', 'z', 'y', 'Total', 'x', 'w'],
                         "count_share_Base Shares": [x / 1000 for x in base], 
                         'a': a,
                         'a_diverted': [x * 560 for x in a]},
                          dtype='object')
    
    b = [1, .5291, .3259, .2032, .4709, .2931, .1778]
    base = [1000, 560, 300, 260, 220, 110, 110]
    b_df = pd.DataFrame({'corp': ['Total', 'a', 'a', 'a', 'c', 'c', 'c'],
                         'choice': ['Total', 'Total', 'u', 'v', 'Total', 'z', 'y'],
                         "count_share_Base Shares": [x / 1000 for x in base],
                         'b': b,
                         'b_diverted': [x *220 for x in b]},
                         dtype='object')
    
    c = [1, .6905, .3788, .3117, .3095, .2576, .0519]
    base = [1000, 560, 300, 260, 220, 110, 110]
    c_df = pd.DataFrame({'corp': ['Total', 'a', 'a', 'a', 'b', 'b', 'b'],
                        'choice': ['Total', 'Total', 'u', 'v', 'Total', 'w', 'x'],
                        "count_share_Base Shares": [x / 1000 for x in base],
                        'c': c,
                        'c_diverted': [x * 220 for x in c]},
                         dtype='object')

    answer = {'a': a_df,
              'b': b_df,
              'c': c_df}
    
    for dictionary in [test, answer]:
        for key in ['a', 'b', 'c']:
            df = dictionary[key]
            df["{}".format(key)] = df["{}".format(key)].astype("float")
            df["{}_diverted".format(key)] = df["{}_diverted".format(key)].astype("float")
            df = df.round(decimals=4)
            dictionary[key] = df
            
    
    assert dictionary_comparison(test, answer)
    
def test_ExportDiversions_choice_to_choice(semi_dc, semi_cd_corp, dc_semiparam_corp_divs_choice):
    semi_dc.fit(semi_cd_corp)

    test = semi_dc.export_diversions(dc_semiparam_corp_divs_choice, semi_cd_corp, export=False)
    
    u = [1, .5102, .4592, .051, .3946, .2041, .1905, .0952, .0952]
    base = [1000, 220, 110, 110, 220, 110, 110, 560, 260]
    u_df = pd.DataFrame({'corp': ['Total', 'c', 'c', 'c', 'b', 'b', 'b', 'a', 'a'],
                         'choice': ['Total', 'Total', 'y', 'z', 'Total', 'w', 'x', 'Total', 'v'],
                         "count_share_Base Shares": [x / 1000 for x in base],
                         'u': u,
                         'u_diverted': [x * 300 for x in u]},
                        dtype='object')
    answer = {'u': u_df}    
    
    assert dictionary_comparison(test, answer)

def test_ExportDiversions_OneChoice(semi_dc, semi_cd, dc_semiparam_diversion):
    semi_dc.fit(semi_cd, use_corp=True)

    test = semi_dc.export_diversions(dc_semiparam_diversion[['a']], semi_cd, export=False)
    
    a = [1, .5857, .4143]
    base = [1000, 220, 220]
    answer = {'a': pd.DataFrame({'choice': ['Total', 'c', 'b'],
                                 "count_share_Base Shares": [x / 1000 for x in base],
                                 'a': a,
                                 'a_diverted': [x * 560 for x in a]},
                                dtype='object')}
    
    assert dictionary_comparison(test, answer)

# tests for wtp_change()
@pytest.fixture
def wtp_change():
    actual_df = pd.DataFrame({'a': [912.6297],
                      'b': [287.4548],
                      'combined': [1574.0460],
                      'wtp_change': [.3116]})

    wtp_change = {'trans': actual_df}
    
    return wtp_change
    

def test_wtpchange(semi_cd, wtp_change):
    semi_dc = DiscreteChoice(solver='semiparametric', coef_order = ['x1', 'x2', 'x3'], min_bin=180)

    semi_dc.fit(semi_cd)
    y_hat = semi_dc.predict(semi_cd)  
    
    test = semi_dc.wtp_change(semi_cd, y_hat, {'trans': ['a', 'b']})
    test['trans'] = test['trans'].round(decimals=4)
    
    actual = wtp_change
    
    assert dictionary_comparison(test, actual)

def test_wtpchange_multiple(semi_cd, wtp_change):
    semi_dc = DiscreteChoice(solver='semiparametric', coef_order = ['x1', 'x2', 'x3'], min_bin=180)

    semi_dc.fit(semi_cd)
    y_hat = semi_dc.predict(semi_cd)  
    
    transactions = {'trans': ['a', 'b'],
                    'trans2': ['b', 'c']}
    
    test = semi_dc.wtp_change(semi_cd, y_hat, transactions)
    test['trans'] = test['trans'].round(decimals=4)
    test['trans2'] = test['trans2'].round(decimals=4)

    actual = wtp_change
    actual.update({'trans2': pd.DataFrame({'b':  [287.4548],
                      'c': [252.4201],
                      'combined': [710.9841],
                      'wtp_change': [.3169]})})
    
    assert dictionary_comparison(test, actual)
    
def test_wtpchange_wght(semi_cd_wght, wtp_change):
    semi_dc = DiscreteChoice(solver='semiparametric', coef_order = ['x1', 'x2', 'x3'], min_bin=180)

    semi_dc.fit(semi_cd_wght)
    choice_probs = semi_dc.predict(semi_cd_wght)  
    
    test = semi_dc.wtp_change(semi_cd_wght, choice_probs, {'trans':['a', 'b']})
    test['trans'] = test['trans'].round(decimals=4)

    actual = wtp_change
    
    assert dictionary_comparison(test, actual)
    
def test_wtpchange_warning(semi_cd, semi_dc):
    semi_dc.fit(semi_cd)
    y_hat = semi_dc.predict(semi_cd)  
    
    with pytest.warns(RuntimeWarning) as record:
        semi_dc.wtp_change(semi_cd, y_hat, {'trans':['a', 'b']})
    
    assert len(record) == 1
    
    
def test_wtpchange_warning_results(semi_cd, semi_dc):
    semi_dc.fit(semi_cd)
    y_hat = semi_dc.predict(semi_cd)  
    
    with pytest.warns(RuntimeWarning):
        test = semi_dc.wtp_change(semi_cd, y_hat, {'trans':['a', 'b']})
    
    actual = {'trans': pd.DataFrame({'a': [np.inf],
                          'b': [np.inf],
                          'combined': [np.inf],
                          'wtp_change': [np.NaN]})
              }
    assert dictionary_comparison(test, actual)
    
#tests for DiscreteChoice.upp
@pytest.fixture
def upp_results():
    upp_results = pd.DataFrame({"Name": ['a', 'b'],
                              "Price": [100, 50],
                              "Diversion to Other System": [.4143, .5291],
                              "Margin": [.25, .5],
                              "UPP": [.1036, .2646],
                              "Average UPP": [.1490, .1490]})
    return upp_results

@pytest.fixture
def upp_dict1():
    upp_dict1 = {'name': 'a',
        'price' : 100,
        'margin': .25}
    
    return upp_dict1

@pytest.fixture
def upp_dict2():
    upp_dict2 = {'name': 'b',
        'price' : 50,
        'margin': .5}
    
    return upp_dict2

def test_upp(semi_cd, semi_dc, upp_dict1, upp_dict2, upp_results):
    semi_dc.fit(semi_cd)
    choice_probs = semi_dc.predict(semi_cd)
    
    div_shares = semi_dc.diversion(semi_cd, choice_probs, div_choices=['a', 'b'])

    test = semi_dc.upp(semi_cd, upp_dict1, upp_dict2, div_shares)
    
    actual = upp_results
    
    assert test.round(decimals=4).equals(actual)

def test_upp_wght(semi_cd_wght, semi_dc, upp_dict1, upp_dict2, upp_results):
    semi_dc.fit(semi_cd_wght)
    choice_probs = semi_dc.predict(semi_cd_wght)
    
    div_shares = semi_dc.diversion(semi_cd_wght, choice_probs, div_choices=['a', 'b'])
    
    test = semi_dc.upp(semi_cd_wght, upp_dict1, upp_dict2, div_shares)
    
    actual = upp_results

    assert test.round(decimals=4).equals(actual)

def test_upp_corp(semi_cd_corp, semi_dc, upp_dict1, upp_dict2, upp_results):
    semi_dc.fit(semi_cd_corp)
    choice_probs = semi_dc.predict(semi_cd_corp)
    
    div_shares = semi_dc.diversion(semi_cd_corp, choice_probs, div_choices=['a', 'b'])
    
    test = semi_dc.upp(semi_cd_corp, upp_dict1, upp_dict2, div_shares)
    
    actual = upp_results
    
    assert test.round(decimals=4).equals(actual)
