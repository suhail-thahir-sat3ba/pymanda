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
    zips += [5 for x in range(3)] # in 90 psa
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
    if dict1.keys == dict2.keys:
        for key in dict1.keys:
            matches.append(dict1[key].equals(dict2[key])) 
    
    return all(matches)

def test_BaseShares(cd_psa):
    test_shares = cd_psa.calculate_shares()
    actual_shares = {'Base Shares': pd.DataFrame({'corporation':['x', 'x', 'y', 'y', 'z'],
                                                   'choice': ['a', 'b', 'c', 'd', 'e'],
                                                   'share': [.3, .2, .2, .05, .25]})}
    
    assert dictionary_comparison(test_shares, actual_shares)
    
def test_PsaShares(cd_psa):
    psa_test = {'x_0.75': [1,2,3]}
    test_shares = cd_psa.calculate_shares(psa_test)
    
    actual_shares = {'x_0.75': pd.DataFrame({'corporation': ['x', 'x', 'y', 'y', 'z'],
                                            'choice': ['a', 'b', 'c', 'd', 'e'],
                                            'share': [x / 46 for x in [30, 8, 1,3, 4]]})}
    
    assert dictionary_comparison(test_shares, actual_shares)

#test for calculating HHI shares
@pytest.fixture
def base_shares():
    base_shares = pd.DataFrame({'corporation':['x', 'x', 'y', 'y', 'z'],
                                'choice': ['a', 'b', 'c', 'd', 'e'],
                                'share': [.3, .2, .2, .05, .25]})
    return base_shares

@pytest.fixture
def base_hhi():
    base_hhi = [3750.0] 
    return base_hhi
    
@pytest.fixture
def psa_shares():
    psa_shares = pd.DataFrame({'corporation': ['x', 'x', 'y', 'y', 'z'],
                                'choice': ['a', 'b', 'c', 'd', 'e'],
                                'share': [x / 46 for x in [30, 8, 1,3, 4]]})
    return psa_shares

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
    df_alt['other shares'] = df_alt['share']
    df_alt = df_alt.drop(columns="share")
    
    share_tables = {'Base Shares': df_alt}
    
    test_hhis = cd_psa.calculate_hhi(share_tables, share_col="other shares")
    
    actual_hhis = {'Base Shares': base_hhi}
    
    assert test_hhis == actual_hhis
    
def test_HHI_BadSharecol(cd_psa, base_shares, psa_shares):
    df_alt = psa_shares
    df_alt['other shares'] = df_alt['share']
    df_alt = df_alt.drop(columns="share")
    
    share_tables = {'Base Shares': base_shares,
                'x_0.75': df_alt}
    
    with pytest.raises(KeyError):
        cd_psa.calculate_hhi(share_tables)

def test_HHI_BadInput(cd_psa):
    with pytest.raises(TypeError):
        cd_psa.calculate_hhi(cd_psa.data)

# test for calculating changes in HHI
def test_HHIChange(cd_psa, base_shares, psa_shares, base_hhi, psa_hhi):
    share_dict = {'Base Shares': base_shares,
                  'x_0.75': psa_shares}
    test_change = cd_psa.hhi_change(['y', 'z'], share_dict)
    
    actual_change = {'Base Shares': [base_hhi, 5000, 1250],
                     'x_0.75' : [psa_hhi, 7835839010804385 / 1099511627776, 166277750892401 / 1099511627776]} # approximately 7126.66 post and 151.23 change
    
    assert test_change == actual_change

def test_HHIChange_TransCol(cd_psa, base_shares):
    share_dict = {"Base Shares": base_shares}
    
    test_change = cd_psa.hhi_change(['d', 'e'], share_dict, trans_var="choice")
    
    actual_change = {'Base Shares': [2350, 2600, 250]}
    
    assert test_change==actual_change
    
def test_HHIChange_MultipleTrans(cd_psa, base_shares):
    share_dict = {"Base Shares": base_shares}
    
    test_change = cd_psa.hhi_change(['c', 'd', 'e'], share_dict, trans_var="choice")
    
    actual_change = {'Base Shares': [2350, 3800, 1450]}
    
    assert test_change==actual_change

def test_HHIChange_BadList(cd_psa, base_shares):
    share_dict = {"Base Shares": base_shares}
    
    with pytest.raises(ValueError):
        cd_psa.hhi_change(['c', 'd', 'e'], share_dict)

def test_HHIChange_NotList(cd_psa, base_shares):
    share_dict = {"Base Shares": base_shares}

    with pytest.raises(TypeError):
        cd_psa.hhi_change('x', share_dict)
    
def test_HHIChange_BadTransCol(cd_psa, base_shares):
    share_dict = {"Base Shares": base_shares}

    with pytest.raises(KeyError):
        cd_psa.hhi_change(['c', 'd'], share_dict, trans_var='systen')

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

def test_DC_semiparam_fit_minbin(semi_cd):
    dc = DiscreteChoice(solver='semiparametric', coef_order = ['x1', 'x2', 'x3'], min_bin=50)
    dc.fit(semi_cd)
    
    test = dc.coef_.round(decimals=4)
    
    actual = pd.DataFrame({'group': ['0\b0\b2','0\b1','0\b1\b1', '1\b0\b0', '1\b1\b0', '1\b1\b1', 'ungrouped'],
                'a': [.5882, .3333, 0, .7143, .7143, .5556, .5],
                'b': [.1176, .6667, .6667, .2857, 0, .2222, .25],
                'c': [.2941, 0, .3333, 0, .2857, .2222, .25]}) 
    
    assert test.equals(actual)
    
    
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
    
    
def test_DC_semiparm_diversion(semi_dc, semi_cd):
    
    semi_dc.fit(semi_cd)
    choice_probs = semi_dc.predict(semi_cd)  
    test = semi_dc.diversion(semi_cd, choice_probs, div_choices=['a', 'b', 'c'])
    
    actual = pd.DataFrame({'a': [np.NaN, .4143, .5857],
                           'b': [.5291, np.NaN, .4709],
                           'c': [.6905, .3095, np.NaN]},
                          index = ['a', 'b', 'c'])
    
    assert test.round(decimals=4).equals(actual)

def test_DC_semiparm_diversion_2choice(semi_dc, semi_cd):
    
    semi_dc.fit(semi_cd)
    choice_probs = semi_dc.predict(semi_cd)  
    test = semi_dc.diversion(semi_cd, choice_probs, div_choices=['a', 'b'])
    
    actual = pd.DataFrame({'a': [np.NaN, .4143, .5857],
                           'b': [.5291, np.NaN, .4709]},
                          index = ['a', 'b', 'c'])
    
    assert test.round(decimals=4).equals(actual)

def test_DC_semiparam_fit_corp(semi_dc, semi_cd_corp):
    
    semi_dc.fit(semi_cd_corp)
    choice_probs = semi_dc.predict(semi_cd_corp)
    
    test = semi_dc.diversion(semi_cd_corp, choice_probs, div_choices=['a', 'b', 'c'])
    
    actual = pd.DataFrame({'a': [np.NaN, np.NaN, .1143, .3, .2571, .3286],
                           'b': [.3259, .2032, np.NaN, np.NaN, .1778, .2931],
                           'c': [.3788, .3117, .2576, .0519, np.NaN, np.NaN]},
                          index = ['u', 'v', 'w', 'x', 'y', 'z'])
    
    assert test.round(decimals=4).equals(actual)
    
def test_DC_semiparam_fit_div_shares_col(semi_dc, semi_cd_corp):
    
    semi_dc.fit(semi_cd_corp)
    choice_probs = semi_dc.predict(semi_cd_corp)
    
    test = semi_dc.diversion(semi_cd_corp, choice_probs, div_choices=['u'], div_choices_var = semi_cd_corp.choice_var)
    
    actual = pd.DataFrame({'u': [np.NaN, .0952, .2041, .1905, .4592, .051]},
                          index = ['u', 'v', 'w', 'x', 'y', 'z'])
    
    assert test.round(decimals=4).equals(actual)  


    