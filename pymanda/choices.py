import pymanda
import pandas as pd
import numpy as np
import warnings


"""
ChoiceData
---------
A container for a DataFrame that maintains relevant columns for mergers and
acquisitions analyses

"""
class ChoiceData():
    
    """
     Two-dimensional, size-mutable, potentially heterogeneous tabular data.
    
     Data structure also contains labeled axes (rows and columns).
     Arithmetic operations align on both row and column labels. Can be
     thought of as a dict-like container for Series objects. The primary
     pandas data structure.
     
     Additionally, has parameters to identify variables of interest for 
     calculating competition metrics using customer level data
    
     Parameters
     ----------
     data : Non-Empty pandas.core.frame.DataFrame object
     choice_var : String of Column Name
         Column name that identifies the "choice" of each customer  
     corp_var : String of Column Name, default None
         Column Name that identifies a higher level classification of Choice    
     geog_var : String of Column Name, default None
         Column name that identifies a geography for Customer
     wght_var : String of Column Name, default None
         Column name that identifies weight for customer level
    
     Examples
     --------
     Constructing ChoiceData from a DataFrame.
     
     >>> choices = ['a' for x in range(100)]
     
     >>> data = pandas.DataFrame({'choice': choices})
     >>> cd = pymanda.ChoiceData(data, 'choice')
     """
    
    def __init__(
        self,
        data, 
        choice_var,
        corp_var= None,
        geog_var= None,
        wght_var= None):
        
        if corp_var is None:
            corp_var = choice_var
    
        self.params = {'choice_var' : choice_var,
                       'corp_var' : corp_var,
                       'geog_var' : geog_var,
                       'wght_var' : wght_var}
        
        self.data = data
        self.choice_var = choice_var
        self.corp_var = corp_var
        self.geog_var = geog_var
        self.wght_var = wght_var
        
        if type(data) !=  pd.core.frame.DataFrame:
            raise TypeError ('''Expected type pandas.core.frame.DataFrame Got {}'''.format(type(data)))
        
        if data.empty:
            raise ValueError ('''Dataframe is Empty''')
        
        defined_params = [x for x in [choice_var, corp_var, geog_var, wght_var] if x is not None]
        for param in defined_params:
            if param not in data.columns:
                raise KeyError ('''{} is not a column in Dataframe'''.format(param))
        
        for nonull in [choice_var, corp_var]:
            if len(data[data['choice'] == ''].index) != 0:
                raise ValueError ('''{} has missing values'''.format(nonull))
     
    def corp_map(self):
        """
        Utility fuction to map corporation and choices in self.data

        Returns
        -------
        corp_map : pandas.core.frame.DataFrame
            2 column data frame with corporation name and choice names.

        """
        if self.corp_var == self.choice_var:
            raise RuntimeError('''corp_map should only be called when self.corp_var is defined and different than self.choice_var''')
        corp_map = self.data.groupby([self.corp_var, self.choice_var]).count()
        corp_map = corp_map.reset_index()
        corp_map = corp_map[[self.corp_var, self.choice_var]]
        
        return corp_map
    
    def estimate_psa(self, centers, threshold=[.75, .9]):
        """
        Return a dictionary idenftifying geographies in a choice's Primary 
        Service Area (PSA) at a given threshold.

        Dictionary keys are labeled as "{center}_{threshold}" and values are 
        a list of geographies falling within the threshold
        
        Default will use self.corp_var as choice and self.wght_var for count
        
        Parameters
        ----------
        centers: List of choices to find 
            
        threshold : float or list of floats, 
            Threshold or levels of thresholds to find the PSA for each choice 
            in centers. Default calculates 75% and 90% PSAs.

        Returns
        -------
        Dictionary
            Dictionary keys are labeled as "{center}_{threshold}" and 
            values are a list of geographies falling within the threshold
        
        Examples
        --------
        >>> choices = ['a' for x in range(100)]
        
        >>> zips = [1 for x in range(20)]
        >>> zips += [2 for x in range(20)]
        >>> zips += [3 for x in range(20)]
        >>> zips += [4 for x in range(20)]
        >>> zips += [5 for x in range(20)]
        
        >>> data = pd.DataFrame({'choice': choices, 'geography': zips})
        >>> cd = ChoiceData(data, 'choice', geog_var = 'geography')
        >>> cd.estimate_psa(['a'])
        {'a_0.75': [1, 2, 3, 4], 'a_0.9': [1, 2, 3, 4, 5]}
        
        """
        if self.geog_var is None:
            raise KeyError ("geog_var is not defined")
        
        if type(threshold) != list:
            threshold = [threshold]
        
        if type(centers) != list:
            centers = [centers]
        
        for center in centers:
            if not self.data[self.corp_var].isin([center]).any():
                raise ValueError ("{cen} is not in {corp}".format(cen=center, corp=self.corp_var))
        
        for alpha in threshold:
            if type(alpha) != float:
                raise TypeError ('''Expected threshold to be type float. Got {}'''.format(type(alpha)))
            if not 0 < alpha <= 1:
                raise ValueError ('''Threshold value of {} is not between 0 and 1''').format(alpha)
        
        df = self.data.copy(deep=True)
        if self.wght_var is None:
            df['count'] = 1
            weight = 'count'
        else:
            weight = self.wght_var
            
        df = df[[self.corp_var, self.geog_var, weight]]
     
        df = df.groupby([self.corp_var, self.geog_var]).sum().reset_index() #calculate counts by geography
        
        df['group_total'] = df[weight].groupby(df[self.corp_var]).transform('sum')  #get group totals
        df['share'] = df[weight] / df['group_total'] # calculate share
        df = df.groupby([self.corp_var, self.geog_var]).sum()
        df = df.sort_values([self.corp_var, weight], ascending=False)
        df_start = df.groupby(level=0).cumsum().reset_index() 
        
        output_dict = {}
        for alpha in threshold:
            df = df_start
            df['keep'] = np.where(df['share'].shift().fillna(1).replace(1, 0) < alpha, 1, 0)
            
            df = df[(df[self.corp_var].isin(centers)) & (df['keep']==1)]
            
            for center in centers:
                in_psa = list(df[self.geog_var][df[self.corp_var]==center])
                in_psa.sort()
                output_dict.update({"{cen}_{a}".format(cen=center, a=alpha) : in_psa})
                
        return output_dict
        
    def restriction_checks(self, restriction):
        """
        Checks for custom restrictions
        """
        
        if type(restriction) != pd.core.series.Series:
            raise TypeError ("Expected type pandas.core.series.Series. Got {}".format(type(restriction)))
        
        if restriction.dtype != np.dtype('bool'):
            raise TypeError ("Expected dtype('bool'). Got {}".format(restriction.dtype))
            
    def restrict_data(self, restriction):
        """
        Restrict the data is self.data using an inline series
        
        Parameters
        ----------
        restriction: pandas.core.series.Series with dtyoe=Boolean
            Boolean series identifying which rows to keep given a restriction
        
        Examples
        --------
        >>> choices = ['a' for x in range(10)]
        
        >>> zips = [1 for x in range(2)]
        >>> zips += [2 for x in range(2)]
        >>> zips += [3 for x in range(2)]
        >>> zips += [4 for x in range(2)]
        >>> zips += [5 for x in range(2)]
        
        >>> data = pd.DataFrame({'choice': choices, 'geography': zips})
        >>> cd = ChoiceData(data, 'choice')
        >>> cd.data
          choice  geography
        0      a          1
        1      a          1
        2      a          2
        3      a          2
        4      a          3
        5      a          3
        6      a          4
        7      a          4
        8      a          5
        9      a          5
        >>> cd.restrict_data(data['geography'] != 5)
        >>> cd.data
          choice  geography
        0      a          1
        1      a          1
        2      a          2
        3      a          2
        4      a          3
        5      a          3
        6      a          4
        7      a          4
        
        """

        self.restriction_checks(restriction)
        
        self.data = self.data[restriction]
        
    def calculate_shares(self, psa_dict=None, weight_var=None, restriction=None):
        """
        Create Share Table of values in self.wght_var by self.choice_var
        
        Parameters
        ----------
        psa_dict: dictionary
            dictionary of lists that contain values to be kept in self.geog_var
            If None, calculates shares for full data.
            
        weight_var : str, Optional
            Column Name in self.data to use as weight. Default is None, 
            weighting every observation equally
        
        restriction: pandas.core.series.Series with dtyoe=Boolean
            Optional restriction to be applied to data before calculating 
            shares

        Returns
        -------
        Dictionary
            keys are the same as psa_dict if given or "Base Shares"
            values are corresponding pandas dataframes of the shares
        
        Examples
        --------
    
        >>> choices = ['a' for x in range(30)]
        >>> choices += ['b' for x in range(20)]
        >>> choices += ['c' for x in range(20)]
        >>> choices += ['d' for x in range(5)]
        >>> choices += ['e' for x in range(25)]
        >>> df = pd.DataFrame({choice" : choices})
        >>> cd = ChoiceData(df, "choice")
        
        >>> cd.calculate_shares()
        {'Base Shares':   choice  share
         0      a   0.30
         1      b   0.20
         2      c   0.20
         3      d   0.05
         4      e   0.25}

        """

        if type(psa_dict) != dict and psa_dict is not None:
            raise TypeError ("Expected type dict. Got {}".format(type(psa_dict)))
            
        if restriction is not None:
            self.restriction_checks(restriction)
        
        if weight_var is None:
            weight_var= self.wght_var
        if weight_var not in self.data.columns and weight_var is not None:
            raise KeyError("{} is not a Column in ChoiceData".format(weight_var))
        
        if psa_dict is None:
            psa_dict= {'Base Shares': []}
            base_shares = True
        else:
            if self.geog_var is None:
                raise ValueError ("geog_var is not defined in ChoiceData")
            elif self.geog_var not in self.data.columns:
                raise KeyError("geog_var is not in ChoiceData")
            base_shares = False
            
        if self.corp_var == self.choice_var:
            group = [self.choice_var]
        else:
            group = [self.corp_var, self.choice_var]
        
        output_dict = {}
        for key in psa_dict.keys():
            df = self.data.copy(deep=True)
            
            redefine_weight=False
            if weight_var is None:
                df['count'] = 1
                weight_var = 'count'
                redefine_weight=True
                
            if restriction is not None:
                df = df[restriction]
                
            if not base_shares:
                for geo in psa_dict[key]:
                     if not df[self.geog_var].isin([geo]).any():
                         raise ValueError ("{g} is not in {col}".format(g=geo, col=self.geog_var)) 
                df = df[df[self.geog_var].isin(psa_dict[key])]

            df = df[group + [weight_var]]
            df_shares = (df.groupby(group).sum() / df[weight_var].sum()).reset_index()
            
            df_shares = df_shares.rename(columns = {weight_var: 'share'})
            output_dict.update({key: df_shares})
            
            if redefine_weight:
                weight_var = None

        return output_dict
    
    def shares_checks(self, df, share_col, data="Data"):
        """
        Checks for columns that are supposed to contain shares

        Parameters
        ----------
        df : pandas.core.frame.DataFrame()
            Dataframe containing share_col
        share_col : str
            Name of column in data frame to chekc.
        data : str, optional
            Name of parameter being checked for Error Message.
            The default is "Data".

        """
        
        if share_col not in df.columns:
            raise KeyError("Column '{}' not in ChoiceData".format(share_col))
        if (df[share_col] < 0).any():
            raise ValueError ("Values of '{col}' in {d} contain negative values".format(col=share_col, d=data))
        if df[share_col].sum() != 1:
            raise ValueError ("Values of '{col}' in {d} do not sum to 1".format(col=share_col, d=data))
    
    def calculate_hhi(self, shares_dict, share_col="share", group_col=None):
        """
        Calculates HHIs from precalculated shares at the corporation level
        
        Parameters
        ----------
        shares_dict: dictionary of share tables
            
        share_col : Column name in dataframe, Optional
            column that holds float of shares. Default is 'share'
        
        group_col: Column name in dataframe, Optional
            column of names to calculate HHI on. Default is self.corp_var

        Returns
        -------
        Dictionary
            keys are the same as shares_dict
            values are hhis
        
        Examples
        --------
        >>> corps = ['x' for x in range(50)]
        >>> corps += ['y' for x in range(25)]
        >>> corps += ['z' for x in range(25)]
        
        >>> choices = ['a' for x in range(30)]
        >>> choices += ['b' for x in range(20)]
        >>> choices += ['c' for x in range(20)]
        >>> choices += ['d' for x in range(5)]
        >>> choices += ['e' for x in range(25)]
        >>> df = pd.DataFrame({"corporation": corps,
                          "choice" : choices})
        >>> cd = ChoiceData(df, "choice", corp_var="corporation")
        
        >>> shares = cd.calculate_shares()
        >>> cd.calculate_hhi(shares)
        {'Base Shares': 3750.0}

        """
        if type(shares_dict) != dict:
            raise TypeError ("Expected type dict. Got {}".format(type(shares_dict)))
        
        if group_col is None:
            group_col = self.corp_var
        elif group_col not in self.data.columns:
            raise KeyError ('''"{}" is not a column in ChoiceData'''.format(group_col))
        output_dict = {}
        for key in shares_dict.keys():
            df = shares_dict[key]
            
            if type(df) != pd.core.frame.DataFrame:
                raise TypeError ('''Expected type pandas.core.frame.DataFrame Got {}'''.format(type(df)))
            
            self.shares_checks(df, share_col, data=key)
            
            df = df.groupby(group_col).sum()
            hhi = (df[share_col] * df[share_col]).sum()*10000
            
            output_dict.update({key: hhi})
            
        return output_dict
    
    def hhi_change(self, trans_list, shares, trans_var=None, share_col="share"):
        """
        Calculates change in Herfindahl-Hirschman Index (HHI) from combining 
        a set of choices.

        Parameters
        ----------
        trans_list : list
            list of choices that will be combined for calculating combined hhi.
        shares : dict
            dictoinary of dataframes of shares to calculate HHIs on.
        trans_var : str, optional
            Column name containging objects in trans_list. The default (None) 
            uses self.corp_var.
        share_col : str, optional
            Column name containing values of share. The default is "share".

        Returns
        -------
        dictionary
            key(s) will match shares parameter
            values will be a list of [pre-merge HHI, post-merge HHI, HHI change].
        """
        
        if type(trans_list) != list:
            raise TypeError ('''trans_list expected list. got {}'''.format(type(trans_list)))
        if len(trans_list) < 2:
            raise ValueError ('''trans_list needs atleast 2 elements to compare HHI change''')
            
        if trans_var is None:
            trans_var = self.corp_var

        for elm in trans_list:
            if not self.data[trans_var].isin([elm]).any():
                raise ValueError ('''{element} is not an element in column {col}'''.format(element=elm, col=trans_var))
        
        output_dict = {}
        for key in shares.keys():
            df = shares[key]
            self.shares_checks(df, share_col, data=key)
            
            if trans_var not in df.columns:
                raise KeyError ('''{var} is not column name in {data}'''.format(var=trans_var, data=key))    
            
            pre_hhi = self.calculate_hhi({"x" : df}, share_col, group_col=trans_var)['x']
            
            post_df = df
            post_df[trans_var] = post_df[trans_var].where(~post_df[trans_var].isin(trans_list), 'combined')
            post_hhi = self.calculate_hhi({"x": post_df}, share_col, group_col=trans_var)['x']
            
            hhi_change = post_hhi - pre_hhi
            output_dict.update({key : [pre_hhi, post_hhi, hhi_change]})
            
        return output_dict
      

class DiscreteChoice():
    """
    
    DiscreteChoice
    ---------
    A solver for estimating discrete choice models and post-estimation 
    analysis.     

    Parameters
    ----------
    solver: str of solver to use
    copy_x: Bool, Optional
        whether to create copies of data in calculations. Default is True.
    coef_order: list, Optional
        coefficient order used for solver 'semiparametric'
    verbose: Boolean, Optional
        Verbosity in solvers. Default is False
    min_bin: int or float, Optional
        Minimum bin size used for solver 'semiparametric'
    
    Examples
    --------
    
    DiscreteChoice(solver='semiparametric', coef_order = ['x1', 'x2', 'x3'])

    """
    def __init__(
        self,
        solver='semiperametric', 
        copy_x=True,
        coef_order= None,
        verbose= False,
        min_bin= 25):
        
        self.params = {'solver' : solver,
                       'copy_x' : True,
                       'coef_order' : coef_order,
                       'verbose': verbose,
                       'min_bin': min_bin}
        
        self.solver = solver
        self.copy_x = copy_x
        self.coef_order = coef_order
        self.verbose = verbose
        self.min_bin = min_bin
        
        current_solvers = ['semiparametric']
        if solver not in current_solvers:
            raise ValueError ('''{a} is not supported solver. Solvers currently supported are {b}'''.format(a=solver, b=current_solvers))
        
        if type(copy_x) is not bool:
            raise ValueError ('''{} is not bool type.'''.format(copy_x))
        
        if type(coef_order) != list:
            raise ValueError ('''coef_order expected to be list. got {}'''.format(type(coef_order)))
        if len(coef_order) ==0:
            raise ValueError ('''coef_order must be a non-empty list''')
        
        if type(verbose) is not bool:
            raise ValueError ('''{} is not bool type.'''.format(verbose))
        
        if type(min_bin) != float and type(min_bin) != int:
            raise ValueError('''min_bin must be a numeric value greater than 0''')
        if min_bin <= 0:
            raise ValueError('''min_bin must be greater than 0''') 
    
    def check_is_fitted(self):
        """
        Check that an Instance has been fitted

        """
        
        try:
            self.coef_
        except AttributeError:
            raise RuntimeError('''Instance of DiscreteChoice is not fitted''')
    
    def fit(self, cd, use_corp=False):
        """
        Fit Estimator using ChoiceData and specified solver

        Parameters
        ----------
        cd : pymanda.ChoiceData
            Contains data to be fitted using DiscreteChoice
        use_corp: Boolean
            Whether to fit using corp_var as choice. Default is False.

        """
        
        # if type(cd) !=  pymanda.ChoiceData:
        #     raise TypeError ('''Expected type pymanda.choices.ChoiceData Got {}'''.format(type(cd)))
            
        for coef in self.coef_order:
            if coef not in cd.data.columns:
                raise KeyError ('''{} is not a column in ChoiceData'''.format(coef))
                
        if use_corp:
            choice= cd.corp_var
        else:
            choice= cd.choice_var

        # currently only supports 'semiparametric' solver. Added solvers should use elif statement
        if self.solver=='semiparametric':
            
            X = cd.data[self.coef_order + [choice]].copy()
            
            if cd.wght_var is not None:
                X['wght'] = cd.data[cd.wght_var]
            else:
                X['wght'] = 1
                
            ## group observations
            X['grouped'] = False
            X['group'] = ""
            for i in range(4, len(self.coef_order)+4):
                bin_by_cols = X.columns[0:-i].to_list()
                if self.verbose:
                    print(bin_by_cols)
                
                screen = X[~X['grouped']].groupby(bin_by_cols).agg({'wght':['sum']})
                    
                screen = (screen >= self.min_bin)
                screen.columns = screen.columns.droplevel(0)
                
                X = pd.merge(X, screen, how='left', left_on=bin_by_cols,right_index=True)
                X['sum'] = X['sum'].fillna(True)
                # update grouped and group
                X['group'] = np.where((~X['grouped']) & (X['sum']),
                                      X[bin_by_cols].astype(str).agg('\b'.join,axis=1), X['group'])
                X['grouped'] = X['grouped'] | X['sum']
                X = X.drop('sum', axis=1) 
                
            # group ungroupables
            X.loc[X['group']=="",'group'] = "ungrouped"
            
            # converts from observations to group descriptions
            X = X[[choice] + ['group', 'wght']].pivot_table(index='group', columns=choice, aggfunc='sum', fill_value=0)
            
            #convert from counts to shares
            X['rowsum'] = X.sum(axis=1)
            for x in X.columns:
                X[x] = X[x] / X['rowsum']
            X = X.drop('rowsum', axis=1)
            X.columns = [col[1] for col in X.columns]
            X= X.reset_index()
            
            self.coef_ = X
    
    def predict(self, cd):
        """
        Use Estimated model to predict individual choice

        Parameters
        ----------
        cd : pymanda.ChoiceData
            ChoiceData to be predicted on.

        Returns
        -------
        choice_probs : pandas.core.frame.DataFrame
            Dataframe of predictions for each choice.
            When solver ='semiparametric', each row contains probabilities of 
            going to any of the choices.

        """
        # if type(cd) !=  pymanda.ChoiceData:
        #     raise TypeError ('''Expected type pymanda.choices.ChoiceData Got {}'''.format(type(cd)))

        self.check_is_fitted()
        
        if self.solver == 'semiparametric':
            
            #group based on groups
            X = cd.data[self.coef_order].copy()
            X['group'] = ""
            for n in range(len(self.coef_order)):
                X['g'] = X[self.coef_order[:len(self.coef_order) - n]].astype(str).agg('\b'.join,axis=1)
                X['group'] = np.where((X['g'].isin(self.coef_['group'])) & (X['group'] == ""),
                                      X['g'],
                                      X['group'])
            
            X.loc[X['group']=="",'group'] = "ungrouped"
            X = X['group']
            
            choice_probs = pd.merge(X, self.coef_, how='left', on='group')
            choice_probs = choice_probs.drop(columns=['group'])            
            
        
        return choice_probs

    def diversion(self, cd, choice_probs, div_choices, div_choices_var=None):
        '''
        Calculate diversions given a DataFrame of observations with diversion
        probabilities

        Parameters
        ----------
        cd: pymanda.ChoiceData
            ChoiceData to calculate diversions on.

        choice_probs : pandas.core.frame.DataFrame
            DataFrame of observations with diversion probabilities.
            
        div_choices : list
            list of choices to calculate diversions for.

        Returns
        -------
        div_shares : pandas.core.frame.DataFrame
            Columns are name of choice being diverted, 
            rows are shares of diversion.

        '''
        # if type(cd) !=  pymanda.ChoiceData:
        #     raise TypeError ('''Expected type pymanda.choices.ChoiceData Got {}'''.format(type(cd)))
        
        if type(div_choices) != list and div_choices is not None:
            raise TypeError('''choices is expected to be list. Got {}'''.format(type(div_choices)))
        if len(div_choices) == 0:
            raise ValueError ('''choices must have atleast a length of 1''')
            
        if type(choice_probs) != pd.core.frame.DataFrame:
            raise TypeError ('''Expected Type pandas.core.frame.DataFrame. Got {}'''.format(type(choice_probs)))
        
        if div_choices_var is None:
            choice = cd.choice_var
            if cd.corp_var != cd.choice_var:
                corp_map = cd.corp_map()
        elif div_choices_var not in cd.data.columns:
            raise KeyError("""div_choices_var not in cd.data""")
        else:
            choice = div_choices_var
                    
        if len(choice_probs) != len(cd.data):
            raise ValueError('''length of choice_probs and cd.data should be the same''')
            
        choice_probs['choice'] = cd.data[choice]
        
        all_choices = list(choice_probs['choice'].unique())
            
        for c in all_choices:
            if c not in choice_probs.columns:
                raise KeyError ('''{} is not a column in choice_probs'''.format(c))
        
        if cd.wght_var is not None:
            choice_probs['wght'] = cd.data[cd.wght_var]
        else:
            choice_probs['wght'] = 1
        
        div_shares = pd.DataFrame(index=all_choices)
        for diversion in div_choices:
            if div_choices_var is None and cd.corp_var != cd.choice_var:
                div_list = list(corp_map[corp_map[cd.corp_var].isin([diversion])][cd.choice_var])
            else:
                div_list = [diversion]
                
            df = choice_probs[choice_probs['choice'].isin(div_list)].copy()
            
            all_choice_temp = all_choices.copy()
            all_choice_temp = [x for x in all_choice_temp if x not in div_list]
            
            df = df[all_choice_temp + ['wght']]
            df['rowsum'] = df[all_choice_temp].sum(axis=1)
            for x in all_choice_temp:
                df[x] = df[x] / df['rowsum'] * df['wght']
            df = df.drop(columns=['rowsum', 'wght'])
            df = df.sum()
            df = df / df.sum()

            df.name = diversion
            div_shares = div_shares.merge(df, how='left', left_index=True, right_index=True)    
        
        return div_shares
    
    def wtp_change(self, cd, choice_probs, trans_list):
        """
        Calculate the change in Willingness to Pay (WTP) for a combined entity
        given a DataFrame of predictions

        Parameters
        ----------
        cd: pymanda.ChoiceData
            ChoiceData corresponding to choice_probs.
        choice_probs : pandas.core.frame.DataFrame
            DataFrame of observations with diversion probabilities 
            corresponding to cd.
        trans_list: list
            List of choices to calculate WTP change on.

        Returns
        -------
        wtp_df : pandas.core.frame.DataFrame
            1 row dataframe with columns showing individual WTP of elements 
            in trans_list and wtp of a combined entity

        """
        
        if type(trans_list) != list:
            raise TypeError ('''trans_list expected type list. got {}'''.format(type(trans_list)))
        
        if len(trans_list) < 2:
            raise ValueError ('''trans_list needs atleast 2 choices''')
        
        for tran in trans_list:
            if tran not in choice_probs.columns:
                raise KeyError ('''{} is not a choice in choice_probs'''.format(tran))

        wtp_df = choice_probs[trans_list].copy()
        
        wtp_df['combined'] = wtp_df.sum(axis=1)
        

        if (wtp_df==1).any().any():
            warnings.warn('''A diversion probability for a bin equals 1 which will result in infinite WTP.''' , RuntimeWarning)
        
        with warnings.catch_warnings(record = True): # prevents redundant warning for np.log(0)
            wtp_df = -1 * np.log(1- wtp_df) # -1 * ln(1-prob)
        
        if cd.wght_var is not None:
            cols = wtp_df.columns
            wtp_df['wght'] = cd.data[cd.wght_var]
            for c in cols:
                wtp_df[c] = wtp_df[c] * wtp_df['wght']
            wtp_df = wtp_df.drop(columns=['wght'])
            
        wtp_df = wtp_df.sum().to_frame().transpose()
        
        wtp_df['wtp_change'] = (wtp_df['combined'] - wtp_df[trans_list].sum(axis = 1)) /  wtp_df[trans_list].sum(axis = 1)
    
        return wtp_df        
    
    def upp(self, cd, upp_dict1, upp_dict2, div_shares):
        """
        Calculate Upward Pricing Pressure (UPP)  using estimated diversions for
        2 choices

        Parameters
        ----------
        cd : pymanda.ChoiceData
            ChoiceData corresponding to div_shares.
        upp_dict1 : dict
            Dictionary containing data for necessary UPP calculations.
                'name': name of choice. Always looks at cd.corp_var
                'price': Average corp price
                'margin': Corp margins
        upp_dict2 : dict
            Dictionary containing data for necessary UPP calculations.
                'name': name of choice. Always looks at cd.corp_var
                'price': Average corp price
                'margin': Corp margins
        div_shares : pandas.core.frame.DataFrame
            Diversion shares for choices in upp_dict.

        Returns
        -------
        upp : pandas.core.frame.DataFrame
            Dataframe showing the UPP for each choice and average upp.

        """
        # if type(cd) !=  pymanda.ChoiceData:
        #     raise TypeError ('''Expected type pymanda.choices.ChoiceData Got {}'''.format(type(cd)))
        
        expected_keys = ['name', 'price', 'margin']
        msg = '''upp_dict is expected to have only the following keys: {}'''.format(expected_keys)
        for d in [upp_dict1, upp_dict2]:
            if type(d) != dict:
                raise TypeError('''upp_dict is expected to be type dict. Got {}'''.format(type(d)))
            
            if len(d.keys()) != len(expected_keys):
                raise ValueError(msg)
            for key in d.keys():
                if key not in expected_keys:
                    raise KeyError(msg)
            
                if d['name'] not in list(cd.data[cd.corp_var]):
                    raise KeyError('''{name} is not a choice in ChoiceData column {col}'''.format(name=d['name'], col=cd.data[cd.corp_var]))
                              
                if type(d['price']) not in [int, float]:
                    raise TypeError(''' 'price' is expected to be numeric. Got {}'''.format(type(d['price'])))
                
                if type(d['margin']) not in [int, float]:
                    raise TypeError(''' 'margin' is expected to be numeric. Got {}'''.format(type(d['margin'])))
        
        if type(div_shares) != pd.core.frame.DataFrame:
            raise TypeError('''div_shares expected to be type pandas.core.frame.DataFrame. Got {}'''.format(type(div_shares)))
        
        if cd.corp_var != cd.choice_var:
            corp_map = cd.corp_map()
            index_keep1 = list(corp_map[corp_map[cd.corp_var] == upp_dict1['name']]['choice'])
            index_keep2 = list(corp_map[corp_map[cd.corp_var] == upp_dict2['name']]['choice'])
        else:
            index_keep1 = [upp_dict1['name']]
            index_keep2 = [upp_dict2['name']]

        div1_to_2 = div_shares[upp_dict1['name']][div_shares.index.isin(index_keep2)].sum()
        div2_to_1 = div_shares[upp_dict2['name']][div_shares.index.isin(index_keep1)].sum()
        
        upp1 = div1_to_2 * upp_dict2['margin'] * upp_dict2['price'] / upp_dict1['price']
        upp2 = div2_to_1 * upp_dict1['margin'] * upp_dict1['price'] / upp_dict2['price']        
        
        if cd.wght_var is not None:
            obs1 = cd.data[cd.wght_var][cd.data[cd.corp_var] == upp_dict1['name']].sum()
            obs2 = cd.data[cd.wght_var][cd.data[cd.corp_var] == upp_dict2['name']].sum()
        else:
            obs1 = cd.data[cd.corp_var][cd.data[cd.corp_var] == upp_dict1['name']].count()
            obs2 = cd.data[cd.corp_var][cd.data[cd.corp_var] == upp_dict2['name']].count()
            
        avg_upp = (upp1 * obs1  + upp2 * obs2) / (obs1 + obs2)
        
        upp = pd.DataFrame({'upp_{}'.format(upp_dict1['name']): upp1,
                            'upp_{}'.format(upp_dict2['name']): upp2,
                            'avg_upp': avg_upp}, index = [0])
        
        return upp
