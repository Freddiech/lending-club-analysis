from __future__ import print_function
import pandas as pd
import collections
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import math
import warnings

#geo package
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

class EDA:
    # state nane and abbrevation for geo visualization
    state_abbr = {"Alabama": "AL", "Alaska": "AK", "American Samoa": "AS", "Arizona": "AZ", "Arkansas": "AR", "California": "CA","Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District Of Columbia": "DC","Federated States Of Micronesia": "FM", "Florida": "FL", "Georgia": "GA", "Guam": "GU", "Hawaii": "HI","Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY","Louisiana": "LA", "Maine": "ME", "Marshall Islands": "MH", "Maryland": "MD", "Massachusetts": "MA","Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE","Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY","North Carolina": "NC", "North Dakota": "ND", "Northern Mariana Islands": "MP", "Ohio": "OH", "Oklahoma": "OK","Oregon": "OR", "Palau": "PW", "Pennsylvania": "PA", "Puerto Rico": "PR", "Rhode Island": "RI","South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT","Virgin Islands": "VI", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI","Wyoming": "WY", "Fed States of Micronesia": "FSM",'District of Columbia': "DC", 'Puerto Rico': "PR"}

    
    def __init__(self, data_path,bucket_number,bucket_limit):
        self.data = pd.read_csv(data_path)
        self.bucket_number=bucket_number
        self.bucket_limit=bucket_limit
        # iv table is used to store the information value for features
        self.iv_table = pd.DataFrame(columns=['feature','dtype','iv'])
        
    def data_preview(self, input=None):
        '''
        Return the feature name, data type, data missing rate, data example, statistics for every columns in a dataframe.
        '''
        # input data
        if input is None:
            input = self.data
        # data type
        df = pd.DataFrame(input.dtypes, columns=["dtypes"])
        # missing rate
        df["Null_Rate(%)"] = (input.isnull().mean() * 100).values
        # data preview first row
        df["Feature_Data_Example"] = input.loc[0].values
        # count, mean, std, min, 25%, 50%, 75%, max
        df = pd.concat([df, input.describe().T], axis=1)
        # set index to number
        df.reset_index(inplace=True)
        df.rename(columns={"index": "feature"}, inplace=True)
        return df.sort_values(by="Null_Rate(%)", ascending=True)

    
    def analysis_pivot(self, index, columns, values, aggfunc, input=None, **kwargs):
        '''
        return a pivot table based on the input variable
        '''
        # input data
        if "input" in kwargs: input =kwargs['input']
        else: input = self.data
        
        # generate pivot table
        pivot = pd.pivot_table(input, index=index, columns=columns, values=values, aggfunc=aggfunc)
        return pivot
    
    
    def analysis_iv(self, x_column, y_column, value_column, bad_label, good_label):
        '''
        Binning: Use Equal frequency binning method for continuous variable, set bucket limit for discrete variable
        Pivot：Count the number of positive and negative samples in each bucket
        Information Value: Calculate the information value based on the iv equation
        Return: Return the name, data type, iv of the feature
        '''
        #corner case
        if x_column == y_column: return {'feature': x_column, 'dtype': self.data[x_column].dtypes, 'iv': 'Target Variable'}
        temp = self.data[[x_column, y_column, value_column]]
        try:
            
            #continuous variable: use qcut to set the buckets
            if temp[x_column].dtypes == float:
                temp.loc[:, 'bucket'] = pd.qcut(temp[x_column], self.bucket_number,duplicates='drop')[:]
            
            #discrete variable: if number of category exceed the self.bucket_limit get the top n of most frequent categories
            else:
                if len(pd.unique(temp[x_column])) > self.bucket_limit:
                    count_dict=collections.Counter(list(temp[x_column].map(str)))
                    if "nan" in count_dict: del(count_dict["nan"])
                    df=pd.DataFrame.from_records(list(dict(count_dict).items()), columns=['object','frequency']).sort_values(by='frequency',ascending=False)
                    def classify(input):
                        if input in set(df.head(self.bucket_number)['object'].values): return input
                        else: return "Other"
                    self.data['{}_bucket'.format(x_column)]=self.data[x_column].map(classify)
                    x_column='{}_bucket'.format(x_column)
                    temp = self.data[[x_column, y_column, value_column]]
                temp.loc[:, 'bucket'] = temp[x_column][:]
            
            #use pivot table to calculate the woe,iv_i in each bucket and sum up into final iv
            pivot = pd.pivot_table(temp, index=['bucket'], columns=[y_column], values=[value_column],aggfunc=['count'])
            pivot.loc[:, ('Calculate Field', value_column, 'woe')] = (pivot.loc[:, ('count', value_column, bad_label)].map(lambda x: x / sum(pivot.loc[:, ('count', value_column, bad_label)]))).map(np.log) - (pivot.loc[:, ('count', value_column, good_label)].map(lambda x: x / sum(pivot.loc[:, ('count', value_column, good_label)]))).map(np.log)
            pivot.loc[:, ('Calculate Field', value_column, 'iv')] = pivot.loc[:, ('Calculate Field', value_column, 'woe')] * ((pivot.loc[:,('count',value_column,bad_label)].map(lambda x: x / sum(pivot.loc[:, ('count', value_column, bad_label)]))) - (pivot.loc[:,('count', value_column, good_label)].map(lambda x: x / sum(pivot.loc[:, ('count', value_column, good_label)]))))
            iv=sum(pivot.loc[:,("Calculate Field",value_column,"iv")])
            return {'feature': x_column, 'dtype': self.data[x_column].dtypes, 'iv': iv}
        except:
            return {'feature': x_column, 'dtype': self.data[x_column].dtypes, 'iv': 'Error'}
    
    
    def visualization_plot(self, mode, *args, **kwargs):
        """
        This function can both support return a figure or a subplot
        It use mode variable to control the type of the plot （line, bar）
        args=[[x1,y1,label1],[x2,y2,label2],.....]
        kwargs=[figsize=v1, barwidth=v2,......]
        """
        if "input_plot" in kwargs: ax1 = kwargs["input_plot"]
        else:
            if "figsize" in kwargs: fig, ax1 = plt.subplots(figsize=kwargs["figsize"])
            else: fig, ax1 = plt.subplots()

        for idx, v in enumerate(args):
            x_var = v[0]
            y_var = v[1]
            label = v[2]
            x = np.arange(len(y_var))
            if "barwidth" in kwargs: barwidth = kwargs["barwidth"]
            else: barwidth = 0.35
            
            #use mode to determine the kind of plot
            if mode == "bar": ax1.bar(x + barwidth * idx, y_var, width=barwidth, label=label)
            elif mode == "line": ax1.plot(x + barwidth * len(args) / 2, y_var, label=label)
        
        title = kwargs["title"] if "title" in kwargs else ""
        xtick_rotation = kwargs["rotation"] if "rotation" in kwargs else 0
        if 'legend_location' in kwargs: ax1.legend(loc=kwargs['legend_location'])
        ax1.set_title(title)
        ax1.set_xticks(x + barwidth * (idx+1) / 2)
        ax1.set_xticklabels(list(x_var), rotation=xtick_rotation)
        
        if "input_plot" in kwargs: return ax1
        else: return fig

        
    def visualization_classfied_default_rate(self, feature, target_column, value_column, function,bad_label,good_label,**kwargs):
#         #"""
#         This function rely on the visualization_plot function.
#         It plots the bar chart for the positive and negative sample in every bucket.
#         Also, it plots the line chart for the negative rate in every bucket.
#         The result is returned in one figure.
#         """
        #continuous variable: use qcut to set the buckets
        if self.data[feature].dtypes==float:
            self.data["{}_bucket".format(feature)]=pd.qcut(self.data[feature],self.bucket_number,duplicates="drop")[:]
            feature="{}_bucket".format(feature)
        
        #use pivot table to calculate the negative rate in each bucket
        pivot=self.analysis_pivot(index=[feature], columns=[target_column], values=[value_column], aggfunc=[function])
        pivot.loc[:,('Calculate Field',value_column,"Default Rate")]=pivot.loc[:,(function,value_column,bad_label)]/(pivot.loc[:,(function,value_column,bad_label)]+pivot.loc[:,(function,value_column,good_label)])

        if "figsize" in kwargs: fig, ax1 = plt.subplots(figsize=kwargs['figsize'])
        else: fig, ax1 = plt.subplots()
        if "rotation" in kwargs: xtick_rotation = kwargs["rotation"]
        else: xtick_rotation = 0
        
        # subplot1
        self.visualization_plot("bar",
                               [pivot.index.values,pivot.loc[:,(function,value_column,bad_label)],bad_label],
                               [pivot.index.values,pivot.loc[:,(function,value_column,good_label)],good_label],
                               rotation=xtick_rotation,
                               input_plot=ax1)
        
        #get iv value for the feature to assign the title
        if feature.replace("_bucket","") in self.iv_table["feature"].values:
            titl='Loan_status Distribution and Default Rate \n Divide by {} \n IV( {} )={}'.format(feature,feature,format(float(self.iv_table[self.iv_table["feature"]==feature.replace("_bucket","")]["iv"]),"0.2f"))
        elif "iv_feature" not in kwargs:
            titl='Loan_status Distribution and Default Rate \n Divide by {}'.format(feature,feature)
        elif "iv_feature" in kwargs: 
            titl='Loan_status Distribution and Default Rate \n Divide by {} \n IV( {} )={}'.format(feature,feature,format(float(self.iv_table[self.iv_table["feature"]==kwargs["iv_feature"]]["iv"]),"0.2f"))
        
        # subplot2
        ax2 = ax1.twinx()
        self.visualization_plot("line",
                               [pivot.index.values,pivot.loc[:,('Calculate Field',value_column,"Default Rate")],"Default Rate"],
                               rotation=xtick_rotation,
                               title=titl,
                               input_plot=ax2)
        return pivot,plt

    
    def visualization_geo(self,input_pivot,target,**kwargs):
        '''
        This function returns a map based on input variable and shapefile.
        It uses different shade of color to represent the value in differnet area, which can reflect the spatial distribution of a feature.
        '''
        if "figsize" in kwargs: plt.figure(figsize=kwargs["figsize"])
        else: plt.figure()
        
        #determine the map boundary
        m = Basemap(llcrnrlon=kwargs["llcrnrlon"], llcrnrlat=kwargs["llcrnrlat"], urcrnrlon=kwargs["urcrnrlon"], urcrnrlat=kwargs["urcrnrlat"],
                    projection=kwargs["projection"], lat_1=kwargs["lat_1"], lat_2=kwargs["lat_2"], lon_0=kwargs["lon_0"])   
        
        #import shapefile
        shp_info = m.readshapefile(kwargs["geo_file_path"], 'states', drawbounds=True)
        
        #determine color for each area by the input_pivot
        colors = {}
        statenames = []
        cmap = plt.cm.hot
        vmin = min(input_pivot.loc[:, target])*0.8
        vmax = max(input_pivot.loc[:, target])*1.2
        for shapedict in m.states_info:
            statename = shapedict['NAME']
            if EDA.state_abbr[statename] not in set(input_pivot.index): val = 0
            else: val = input_pivot.loc[EDA.state_abbr[statename], target]
            c=cmap(np.abs((val - vmin) / (vmax - vmin)))
            if type(c)==tuple: colors[statename] = c[:3]
            else: colors[statename] = c[0][:3]
            statenames.append(statename)
        
        #plot the color on map
        ax = plt.gca()
        for nshape, seg in enumerate(m.states):
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax.add_patch(poly)
            
        if "title" in kwargs: plt.title(kwargs['title'])
        
        #set map legend
        tmp = np.linspace(vmin, vmax, 100)
        im = plt.imshow(np.array([tmp, tmp]), cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        return plt
