import pandas as pd
import numpy as np

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
            
        threshold : float or list of floats, default [.75, .9]
            Threshold or levels of thresholds to find the PSA for each choice 
            in centers

        Returns
        -------
        Dictionary
        
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
        
        
        if self.wght_var is None:
            self.data['count'] = 1
            self.wght_var = "count"
        
        df = self.data[[self.corp_var, self.geog_var, self.wght_var]]
     
        df = df.groupby([self.corp_var, self.geog_var]).sum().reset_index() #calculate counts by geography
        
        df['group_total'] = df[self.wght_var].groupby(df[self.corp_var]).transform('sum')  #get group totals
        df['share'] = df[self.wght_var] / df['group_total'] # calculate share
        df = df.groupby([self.corp_var, self.geog_var]).sum()
        df = df.sort_values([self.corp_var, self.wght_var], ascending=False)
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
    
    def restrict_data(self, restriction):
        """
        Restrict the data is self.data using an inline series
        
        Parameters
        ----------
        restriction: pandas.core.series.Series with Boolean Date Type
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
        if type(restriction) != pd.core.series.Series:
            raise TypeError ("Expected type pandas.core.series.Series. Got {}".format(type(restriction)))
        
        if restriction.dtype != np.dtype('bool'):
            raise TypeError ("Expected dtype('bool'). Got {}".format(restriction.dtype))
        
        self.data = self.data[restriction]
        