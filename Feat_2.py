import pandas as pd
import numpy as np
import numpy
import random
import scipy.special as special
import datetime
class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)

class Feat:
    def FeatSection(self,f,base):
        '特征区间'

        # '先加入所有基础属性的均值特征'

        # '每个商店基础数据的均值特征'
        # tmp = f[['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', \
        #          'shop_star_level', 'shop_score_service','shop_score_delivery', 'shop_score_description']]
        # tmp = pd.pivot_table(tmp, index=['shop_id'], \
        #                      values=['shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', \
        #                              'shop_score_service', 'shop_score_delivery', 'shop_score_description'], \
        #                      aggfunc=[np.mean], fill_value=0)
        #
        # base = pd.merge(base, tmp.reset_index(), how='left', on='shop_id')
        #
        # '广告属性的均值特征'
        # tmp = f[['item_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']]
        # tmp = pd.pivot_table(tmp, index=['item_id'], \
        #                      values=['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level'], \
        #                      aggfunc=[np.mean], fill_value=0)
        # base = pd.merge(base, tmp.reset_index(), on='item_id', how='left')
        #
        # '用户属性的均值特征'
        # tmp = f[['user_id', 'user_age_level', 'user_star_level']]
        # tmp = pd.pivot_table(tmp, index=['user_id'], values=['user_age_level', 'user_star_level'], \
        #                      aggfunc=[np.mean], fill_value=0)
        # base = pd.merge(base, tmp.reset_index(), on='user_id', how='left')

        '商店的点击与消费次数'
        tmp = f[['shop_id', 'is_trade']]
        grouped_trade = pd.pivot_table(tmp,index='shop_id',values='is_trade',aggfunc='count').reset_index()
        grouped_trade.columns = ['shop_id', 'shop_trade_sum']
        tmp = tmp[tmp['is_trade'] == 1]
        grouped_traded = pd.pivot_table(tmp,index='shop_id',values='is_trade',aggfunc='count').reset_index()
        grouped_traded.columns = ['shop_id', 'shop_trade_count']
        shop_trade = pd.merge(grouped_trade, grouped_traded, on='shop_id', how='left')
        shop_trade = shop_trade.fillna(0)
        base = pd.merge(base, shop_trade, on='shop_id', how='left')
        base = base.fillna(0)

        '广告的点击与消费次数'
        tmp = f[['item_id', 'is_trade']]
        item_trade_sum = pd.pivot_table(tmp,index='item_id',values='is_trade',aggfunc='count').reset_index()
        item_trade_sum.columns = ['item_id', 'item_trade_sum']
        base = pd.merge(base, item_trade_sum, on='item_id', how='left')
        tmp=tmp[tmp['is_trade']==1]
        item_trade_count=pd.pivot_table(tmp,index='item_id',values='is_trade',aggfunc='count').reset_index()
        item_trade_count.columns=['item_id','item_trade_count']
        base=pd.merge(base,item_trade_count,on='item_id',how='left')
        base = base.fillna(0)

        '用户的点击与消费次数'
        tmp = f[['user_id', 'is_trade']]
        user_trade_sum = pd.pivot_table(tmp,index='user_id',values='is_trade',aggfunc='count').reset_index()
        user_trade_sum.columns = ['user_id', 'user_trade_sum']
        tmp = tmp[tmp['is_trade'] == 1]
        user_trade_count = pd.pivot_table(tmp,index='user_id',values='is_trade',aggfunc='count').reset_index()
        user_trade_count.columns = ['user_id', 'user_trade_count']
        user_trade = pd.merge(user_trade_sum, user_trade_count, on='user_id', how='left')
        base = pd.merge(base, user_trade, on='user_id', how='left')
        base = base.fillna(0)

        '广告的用户数量'
        ad_user = f[['user_id', 'item_id', 'is_trade']]
        grouped = pd.pivot_table(ad_user,index=['user_id','item_id'],values='is_trade',aggfunc='count').reset_index()
        grouped.columns = ['user_id', 'item_id', 'ad_user_count']
        base = pd.merge(base, grouped, on=['user_id', 'item_id'], how='left')
        base = base.fillna(0)

        '广告用户的消费量'
        ad_user_trade = ad_user[ad_user['is_trade'] == 1]
        grouped = pd.pivot_table(ad_user_trade,index=['user_id','item_id'],values='is_trade',aggfunc='count').reset_index()
        grouped.columns = ['user_id', 'item_id', 'ad_user_trade']
        base = pd.merge(base, grouped, on=['user_id', 'item_id'], how='left')
        base = base.fillna(0)

        '将item_category_list第二个或者第三个作为key，提取点击成交数'
        user_cat = f[['user_id', 'item_category_list', 'is_trade']]
        user_cat['cat'] = user_cat['item_category_list'].map(lambda x: x.split(";"))
        user_cat['key'] = user_cat['cat'].map(lambda x: x[1] if len(x) <= 2 else x[2])

        base['cat'] = base['item_category_list'].map(lambda x: x.split(";"))
        base['key'] = base['cat'].map(lambda x: x[1] if len(x) <= 2 else x[2])
        base = base.drop(['item_category_list', 'cat'], axis=1)

        user_cat = user_cat[['user_id', 'key', 'is_trade']]
        user_cat_sum = pd.pivot_table(user_cat,index=['user_id','key'],values='is_trade',aggfunc='count').reset_index()
        user_cat_sum.columns = ['user_id', 'key', 'user_cat_sum']
        base = pd.merge(base, user_cat_sum, on=['user_id', 'key'], how='left')

        user_cat_count = user_cat[user_cat['is_trade'] == 1]
        user_cat_count = pd.pivot_table(user_cat_count,index=['user_id','key'],values='is_trade',aggfunc='count').reset_index()
        user_cat_count.columns = ['user_id', 'key', 'user_cat_count']
        base = pd.merge(base, user_cat_count, on=['user_id', 'key'], how='left')
        base = base.fillna(0)
        base = base.drop(['key'], axis=1)

        '用户的商店数量'
        user_shop = f[['user_id', 'shop_id', 'is_trade']]
        user_shop_sum = pd.pivot_table(user_shop,index=['user_id','shop_id'],values='is_trade',aggfunc='count').reset_index()
        user_shop_sum.columns = ['user_id', 'shop_id', 'user_shop_sum']
        base = pd.merge(base, user_shop_sum, on=['user_id', 'shop_id'], how='left')
        base = base.fillna(0)

        '用户与商店的消费数量'
        user_shop = user_shop[user_shop['is_trade'] == 1]
        user_shop_trade = pd.pivot_table(user_shop,index=['user_id','shop_id'],values='is_trade',aggfunc='count').reset_index()
        user_shop_trade.columns = ['user_id', 'shop_id', 'user_shop_trade']
        base = pd.merge(base, user_shop_trade, on=['user_id', 'shop_id'], how='left')
        base = base.fillna(0)


        '每小时的用户量'
        hour_user = f[['context_timestamp', 'user_id', 'is_trade']]
        def get_hour(x):
            return x.hour

        hour_user['context_timestamp'] = hour_user['context_timestamp'].map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        hour_user['hour'] = hour_user['context_timestamp'].map(get_hour)
        hour_user = hour_user.drop(['context_timestamp'], axis=1)
        '用户的小时总量'
        time_user_sum = pd.pivot_table(hour_user,index=['user_id','hour'],values='is_trade',aggfunc='count').reset_index()
        time_user_sum.columns = ['user_id', 'hour', 'time_user_sum']
        base = pd.merge(base, time_user_sum, on=['user_id', 'hour'], how='left')
        base = base.fillna(0)
        '用户每小时的消费量'
        time_user_count=hour_user[hour_user['is_trade']==1]
        time_user_count=pd.pivot_table(time_user_count,index=['user_id','hour'],values='is_trade',aggfunc='count').reset_index()
        time_user_count.columns=['user_id','hour','time_user_count']
        base=pd.merge(base,time_user_count,on=['user_id','hour'],how='left')
        base=base.fillna(0)

        '每小时的广告量'
        hour_item = f[['context_timestamp', 'item_id', 'is_trade']]
        def get_hour(x):
            return x.hour

        hour_item['context_timestamp'] = hour_item['context_timestamp'].map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        hour_item['hour'] = hour_item['context_timestamp'].map(get_hour)
        hour_item = hour_item.drop(['context_timestamp'], axis=1)
        '广告小时的总量'
        hour_item_sum = pd.pivot_table(hour_item,index=['item_id','hour'],values='is_trade',aggfunc='count').reset_index()
        hour_item_sum.columns = ['item_id', 'hour', 'hour_item_sum']
        base = pd.merge(base, hour_item_sum, on=['item_id', 'hour'], how='left')
        base = base.fillna(0)

        '广告小时的消费量'
        hour_item_count = hour_item[hour_item['is_trade'] == 1]
        hour_item_count = pd.pivot_table(hour_item_count,index=['item_id','hour'],values='is_trade',aggfunc='count').reset_index()
        hour_item_count.columns = ['item_id', 'hour', 'hour_item_count']
        base = pd.merge(base, hour_item_count, on=['item_id', 'hour'], how='left')
        base = base.fillna(0)

        '小时与商户'
        hour_shop = f[['context_timestamp', 'shop_id', 'is_trade']]
        def get_hour(x):
            return x.hour

        hour_shop['context_timestamp'] = hour_shop['context_timestamp'].map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        hour_shop['hour'] = hour_shop['context_timestamp'].map(get_hour)
        hour_shop = hour_shop.drop(['context_timestamp'], axis=1)

        '每小时的商店量'
        hour_shop_sum =pd.pivot_table(hour_shop,index=['shop_id','hour'],values='is_trade',aggfunc='count').reset_index()
        hour_shop_sum.columns = ['shop_id', 'hour', 'hour_shop_sum']
        base = pd.merge(base, hour_shop_sum, on=['shop_id', 'hour'], how='left')
        base = base.fillna(0)

        '每小时的商店消费量'
        hour_shop_count = hour_shop[hour_shop['is_trade'] == 1]
        hour_shop_count = pd.pivot_table(hour_shop_count,index=['shop_id','hour'],values='is_trade',aggfunc='count').reset_index()
        hour_shop_count.columns = ['shop_id', 'hour', 'hour_shop_count']
        base = pd.merge(base, hour_shop_count, on=['shop_id', 'hour'], how='left')
        base = base.fillna(0)

        '小时与用户 商户'
        user_shop_hour = f[['context_timestamp', 'user_id', 'shop_id', 'is_trade']]
        def get_hour(x):
            return x.hour
        user_shop_hour['context_timestamp'] = user_shop_hour['context_timestamp'].map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        user_shop_hour['hour'] = user_shop_hour['context_timestamp'].map(get_hour)
        user_shop_hour = user_shop_hour.drop(['context_timestamp'], axis=1)

        'user_shop与hour的统计次数'
        user_shop_hour_sum = pd.pivot_table(user_shop_hour,index=['user_id','shop_id','hour'],values='is_trade').reset_index()
        user_shop_hour_sum.columns = ['user_id', 'shop_id', 'hour', 'user_shop_hour_sum']
        base = pd.merge(base, user_shop_hour_sum, on=['user_id', 'shop_id', 'hour'], how='left')
        base = base.fillna(0)

        'user_shop与hour的交易次数'
        user_shop_hour_count = user_shop_hour[user_shop_hour['is_trade'] == 1]
        user_shop_hour_count =pd.pivot_table(user_shop_hour_count,index=['user_id','shop_id','hour'],values='is_trade',aggfunc='count').reset_index()
        user_shop_hour_count.columns = ['user_id', 'shop_id', 'hour', 'user_shop_hour_count']
        base = pd.merge(base, user_shop_hour_count, on=['user_id', 'shop_id', 'hour'], how='left')
        base = base.fillna(0)

        '小时与用户 广告'
        user_item_hour = f[['user_id', 'item_id', 'context_timestamp', 'is_trade']]
        def get_hour(x):
            return x.hour
        user_item_hour['context_timestamp'] = user_item_hour['context_timestamp'].map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        user_item_hour['hour'] = user_item_hour['context_timestamp'].map(get_hour)
        user_item_hour = user_item_hour.drop(['context_timestamp'], axis=1)

        'user_item 与 hour的次数'
        user_item_hour_sum = pd.pivot_table(user_item_hour,index=['user_id','item_id','hour'],values='is_trade',aggfunc='count').reset_index()
        user_item_hour_sum.columns = ['user_id', 'item_id', 'hour', 'user_item_hour_sum']
        base = pd.merge(base, user_item_hour_sum, on=['user_id', 'item_id', 'hour'], how='left')
        base = base.fillna(0)

        'user_item 与 hour的交易次数'
        user_item_hour_count = user_item_hour[user_item_hour['is_trade'] == 1]
        user_item_hour_count =pd.pivot_table(user_item_hour_count,index=['user_id','item_id','hour'],values='is_trade',aggfunc='count').reset_index()
        user_item_hour_count.columns = ['user_id', 'item_id', 'hour', 'user_item_hour_count']
        base = pd.merge(base, user_item_hour_count, on=['user_id', 'item_id', 'hour'], how='left')
        base = base.fillna(0)

        return base
    pass
class Statics_Feat:
    def Shop_feat(self,dataset,base):

        # '将序列型数据转换'
        # 'shop_star_level归一化'
        # def f(x):
        #     return x-4998
        # dataset['shop_star_level'] = dataset['shop_star_level'].map(f)

        # '每个商店基础数据的统计特征'
        # tmp = dataset[
        #     ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', \
        #      'shop_score_delivery', 'shop_score_description']]
        # tmp = pd.pivot_table(tmp, index=['shop_id'], \
        #                      values=['shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', \
        #                              'shop_score_service', 'shop_score_delivery', 'shop_score_description'], \
        #                      aggfunc=[np.mean], fill_value=0)
        #
        # base = pd.merge(base, tmp.reset_index(), how='left', on='shop_id')


        '商店的点击与消费次数'
        tmp=dataset[['shop_id','is_trade']]
        grouped_trade=tmp.groupby(by='shop_id').count().reset_index()
        grouped_trade.columns=['shop_id','shop_trade_sum']
        tmp=tmp[tmp['is_trade']==1]
        grouped_traded=tmp.groupby(by=['shop_id']).count().reset_index()
        grouped_traded.columns=['shop_id','shop_trade_count']
        shop_trade=pd.merge(grouped_trade,grouped_traded,on='shop_id',how='left')
        shop_trade=shop_trade.fillna(0)

        # bs = BayesianSmoothing(1, 1)
        # bs.update(shop_trade['shop_trade_sum'].values, shop_trade['shop_trade_count'].values, 1000, 0.001)
        # shop_trade['shop_trade_smooth'] = (shop_trade['shop_trade_count'] + bs.alpha) / (shop_trade['shop_trade_sum'] + bs.alpha + bs.beta)

        # shop_trade=shop_trade.set_index('shop_id')
        # def f(x):
        #     return x[1]/x[0]
        # shop_trade['shop_trade_ratio']=shop_trade.apply(f,axis=1)
        # shop_trade=shop_trade.reset_index()
        # shop_trade=shop_trade.drop(['shop_trade_sum'],axis=1)

        base=pd.merge(base,shop_trade,on='shop_id',how='left')
        # base['shop_trade_ratio']=base['shop_trade_count']/base['shop_trade_sum']
        base=base.fillna(0)
        # base['shop_trade_smooth']=base['shop_trade_smooth'].fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)

        '商店消费次数大于2'
        # tmp=dataset[['shop_id','is_trade']]
        # grouped=tmp.groupby(by='shop_id').count()
        # grouped.columns=['shop_count']
        # grouped['shop_plus_2']=grouped['shop_count'].map(lambda x:1 if x>=2 else 0)
        # grouped=grouped.drop(['shop_count'],axis=1)
        # grouped=grouped.reset_index()
        # base=pd.merge(base,grouped,on='shop_id',how='left')

        '商店的未消费次数及比例'
        # tmp=dataset[['shop_id','is_trade']]
        # tmp=tmp[tmp['is_trade']==0]
        # grouped=tmp.groupby(by='shop_id').count().reset_index()
        # grouped.columns=['shop_id','shop_untrade_count']
        # shop_untrade=pd.merge(grouped_trade[['shop_id']],grouped,on='shop_id',how='left')
        # base=pd.merge(base,shop_untrade,on='shop_id',how='left')
        # base['shop_untrade_ratio']=base['shop_untrade_count']/base['shop_trade_sum']
        # base=base.fillna(-1)

        return base
        pass
    def Ad_feat(self,dataset,base):

        # '基础属性的统计特征'
        # tmp = dataset[['item_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']]
        # tmp = pd.pivot_table(tmp, index=['item_id'], \
        #                      values=['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level'], \
        #                      aggfunc=[np.mean], fill_value=0)
        # base = pd.merge(base, tmp.reset_index(), on='item_id', how='left')

        '广告的消费次数'
        tmp=dataset[['item_id','is_trade']]
        item_trade_sum=tmp.groupby(by='item_id').count().reset_index()
        item_trade_sum.columns=['item_id','item_trade_sum']
        base=pd.merge(base,item_trade_sum,on='item_id',how='left')
        # pos=tmp[tmp['is_trade']==1]
        # pos=pos.groupby(by='item_id').count().reset_index()
        # pos.columns=['item_id','item_trade_count']
        # con=pd.merge(item_trade_sum,pos,on='item_id',how='left')

        # bs=BayesianSmoothing(1, 1)
        # bs.update(con['item_trade_sum'].values,con['item_trade_count'].values,1000,0.001)
        # con['item_trade_smooth']=(con['item_trade_count']+bs.alpha)/(con['item_trade_sum']+bs.alpha+bs.beta)

        # base=pd.merge(base,con,on='item_id',how='left')
        base = base.fillna(0)

        # base['item_trade_smooth']=base['item_trade_smooth'].fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
        # base['item_trade_ratio']=base['item_trade_count']/base['item_trade_sum']


        # '广告未消费次数及比例'
        # tmp = dataset[['item_id', 'is_trade']]
        # neg = tmp[tmp['is_trade'] == 0]
        # neg = neg.groupby(by='item_id').count().reset_index()
        # neg.columns = ['item_id','item_untrade_count']
        # con = pd.merge(item_trade_sum[['item_id']], neg, on='item_id', how='left')
        # base=pd.merge(base,con,on='item_id',how='left')
        # base['item_untrade_ratio']=base['item_untrade_count']/base['item_trade_sum']
        # base=base.fillna(-1)

        # base = pd.merge(base, con[['item_id', 'item_trade_sum', 'item_untrade_count', 'item_untrade_ratio']], on='item_id',
        #                 how='left')

        '城市的消费比例'
        # item_brand_id=dataset[['item_brand_id','is_trade']]
        # grouped_sum=item_brand_id.groupby(by='item_brand_id').count().reset_index()
        # grouped_sum.columns=['item_brand_id','item_brand_sum']
        # tmp=item_brand_id[item_brand_id['is_trade']==1]
        # grouped_count=tmp.groupby(by='item_brand_id').count().reset_index()
        # grouped_count.columns=['item_brand_id','item_brand_count']
        # grouped=pd.merge(grouped_sum,grouped_count,on='item_brand_id',how='left')
        # grouped=grouped.set_index('item_brand_id')
        # def f(x):
        #     return x[1]/x[0]
        # grouped['item_brand_ratio']=grouped.apply(f,axis=1)
        # grouped=grouped.reset_index()
        # base=pd.merge(base,grouped,on='item_brand_id',how='left')

        '品牌的消费比例'
        # item_city_id=dataset[['item_city_id','is_trade']]
        # grouped_sum=item_city_id.groupby(by='item_city_id').count().reset_index()
        # grouped_sum.columns=['item_city_id','item_city_sum']
        # tmp=item_city_id[item_city_id['is_trade']==1]
        # grouped_count=tmp.groupby(by='item_city_id').count().reset_index()
        # grouped_count.columns=['item_city_id','item_city_count']
        # grouped=pd.merge(grouped_sum,grouped_count,on='item_city_id',how='left')
        # grouped=grouped.set_index('item_city_id')
        # grouped['item_city_ratio']=grouped.apply(f,axis=1)
        # grouped=grouped.reset_index()
        # base=pd.merge(base,grouped,on='item_city_id',how='left')

        '广告的消费次数大于5'
        # tmp=dataset[['item_id','is_trade']]
        # tmp=tmp[tmp['is_trade']==1]
        # grouped=tmp.groupby(by='item_id').count()
        # grouped.columns=['item_count']
        # grouped['ad_plus_5']=grouped['item_count'].map(lambda x: 1 if x>=5 else 0)
        # grouped=grouped.drop(['item_count'],axis=1)
        # grouped=grouped.reset_index()
        # base=pd.merge(base,grouped,on='item_id',how='left')

        return base
        pass
    def User_feat(self,dataset,base):
        '基础属性的的统计特征'
        # tmp = dataset[['user_id', 'user_age_level', 'user_star_level']]
        # tmp = pd.pivot_table(tmp, index=['user_id'], values=['user_age_level', 'user_star_level'], \
        #                      aggfunc=[np.mean], fill_value=0)
        # base = pd.merge(base, tmp.reset_index(), on='user_id', how='left')

        '用户的点击与消费次数'
        tmp=dataset[['user_id','is_trade']]
        user_trade_sum=tmp.groupby(by='user_id').count().reset_index()
        user_trade_sum.columns=['user_id','user_trade_sum']
        tmp=tmp[tmp['is_trade']==1]
        user_trade_count=tmp.groupby(by=['user_id']).count().reset_index()
        user_trade_count.columns=['user_id','user_trade_count']
        user_trade=pd.merge(user_trade_sum,user_trade_count,on='user_id',how='left')


        # def f(x):
        #     return x[1]/x[0]
        # user_trade=user_trade.set_index('user_id')
        # user_trade['user_trade_ratio']=user_trade.apply(f,axis=1)
        base=pd.merge(base,user_trade,on='user_id',how='left')
        # base['user_trade_ratio']=base['user_trade_count']/base['user_trade_sum']
        base=base.fillna(0)

        # bs = BayesianSmoothing(1, 1)
        # bs.update(base['user_trade_sum'].values, base['user_trade_count'].values, 100 ,0.001)
        # base['user_trade_smooth'] = (base['user_trade_count'] + bs.alpha) / (base['user_trade_sum'] + bs.alpha + bs.beta)

        # '用户没有消费的比例'
        # tmp=dataset[['user_id','is_trade']]
        # tmp=tmp[tmp['is_trade']==0]
        # user_untrade_count=tmp.groupby(by='user_id').count().reset_index()
        # user_untrade_count.columns=['user_id','user_untrade_count']
        # user_untrade=pd.merge(user_trade_sum[['user_id']],user_untrade_count,on='user_id',how='left')
        # base=pd.merge(base,user_untrade,on='user_id',how='left')
        # base['user_untrade_ratio']=base['user_untrade_count']/base['user_trade_sum']
        # base=base.fillna(-1)


        # '用户消费次数是否大于5'
        # tmp=dataset[['user_id','is_trade']]
        # grouped=tmp.groupby(by='user_id').count()
        # grouped.columns=['user_count']
        # grouped['user_plus_5']=grouped['user_count'].map(lambda x:1 if x>=5 else 0)
        # grouped=grouped.drop(['user_count'],axis=1)
        # grouped=grouped.reset_index()
        # base=pd.merge(base,grouped,on='user_id',how='left')

        return base
        pass

class Business_Feat:
    def User_Ad(self,f,base):
        'User_Ad交互特征'

        'User_Ad属性平均值'
        # base=base[['instance_id','user_id','item_id']]
        # ad_static=ad[['item_id','item_price_level','item_sales_level',\
        #                        'item_collected_level','item_pv_level']]
        # user_ad_statics=pd.concat([user_id,ad_static],axis=1)
        # grouped=user_ad_statics.groupby(by=['user_id','item_id']).mean().reset_index()
        # base=pd.merge(base,grouped,on=['user_id','item_id'],how='left').fillna(0)

        '广告的平均用户年龄----可以考虑'
        # user_age=user[['user_age_level']]
        # item_id=ad[['item_id']]
        # user_ad_age=pd.concat([user_age,item_id],axis=1)
        # grouped=user_ad_age.groupby(by=['item_id']).mean().reset_index()
        # base=pd.merge(base,grouped,on=['item_id'],how='left')
        # base=base.fillna(-1)


        '消费广告与职业的转化率'
        # user_occupation_id=user[['user_occupation_id']]
        # user_ad_trade=pd.concat([user_occupation_id,ad[['item_id','is_trade']]],axis=1)
        # grouped_sum=user_ad_trade.groupby(by=['item_id','user_occupation_id']).count().unstack().reset_index()
        # user_ad_trade=user_ad_trade[user_ad_trade['is_trade']==1]
        # grouped_traded=user_ad_trade.groupby(by=['item_id','user_occupation_id']).count().unstack().reset_index()
        # grouped_traded=pd.merge(item_id,grouped_traded,on='item_id',how='left')
        # grouped_traded=grouped_traded.set_index('item_id')
        # grouped_sum=grouped_sum.set_index('item_id')
        # tmp=grouped_traded/grouped_sum
        # tmp=tmp.reset_index()
        # tmp=tmp.drop_duplicates(['item_id'])
        # base=pd.merge(base,tmp,on='item_id',how='left')


        '广告的用户数量'
        ad_user=f[['user_id','item_id','is_trade']]
        grouped=ad_user.groupby(by=['user_id','item_id']).count().reset_index()
        grouped.columns=['user_id','item_id','ad_user_count']
        base=pd.merge(base,grouped,on=['user_id','item_id'],how='left')
        base=base.fillna(0)

        '广告用户的消费量'
        ad_user_trade=ad_user[ad_user['is_trade']==1]
        grouped=ad_user_trade.groupby(by=['user_id','item_id']).count().reset_index()
        grouped.columns=['user_id','item_id','ad_user_trade']
        base=pd.merge(base,grouped,on=['user_id','item_id'],how='left')
        base=base.fillna(0)

        '将item_category_list第二个或者第三个作为key，提取点击成交数'
        user_cat = f[['user_id','item_category_list','is_trade']]
        user_cat['cat'] = user_cat['item_category_list'].map(lambda x: x.split(";"))
        user_cat['key'] = user_cat['cat'].map(lambda x: x[1] if len(x) <= 2 else x[2])

        base['cat']=base['item_category_list'].map(lambda x:x.split(";"))
        base['key']=base['cat'].map(lambda x:x[1] if len(x)<=2 else x[2])
        base=base.drop(['item_category_list','cat'],axis=1)

        user_cat=user_cat[['user_id','key','is_trade']]
        user_cat_sum=user_cat.groupby(by=['user_id','key']).count().reset_index()
        user_cat_sum.columns=['user_id','key','user_cat_sum']
        base=pd.merge(base,user_cat_sum,on=['user_id','key'],how='left')

        user_cat_count=user_cat[user_cat['is_trade']==1]
        user_cat_count=user_cat_count.groupby(by=['user_id','key']).count().reset_index()
        user_cat_count.columns=['user_id','key','user_cat_count']
        base=pd.merge(base,user_cat_count,on=['user_id','key'],how='left')
        base = base.fillna(0)

        base=base.drop(['key'],axis=1)


        # '广告的性别数量及比例'
        # user_gender_id=user[['user_gender_id','is_trade']]
        # ad_user=pd.concat([item_id,user_gender_id],axis=1)
        # grouped=ad_user.groupby(by=['item_id','user_gender_id']).count().unstack()
        # grouped.columns=['gender_0','gender_1','gender_2']
        # sum=grouped['gender_0'].sum()+grouped['gender_1'].sum()+grouped['gender_2'].sum()
        # grouped['ad_gender0_ratio']=grouped['gender_0']/sum
        # grouped['ad_gender1_ratio']=grouped['gender_1']/sum
        # grouped['ad_gender2_ratio']=grouped['gender_2']/sum
        # # grouped=grouped.drop(['gender_0','gender_1','gender_2'],axis=1)
        # base=pd.merge(base,grouped.reset_index(),on='item_id',how='left')








        return base
    def User_Shop(self,f,base):
        'User Shop 交互特征'


        '用户的商店数量'
        user_shop=f[['user_id','shop_id','is_trade']]
        user_shop_sum=user_shop.groupby(by=['user_id','shop_id']).count().reset_index()
        user_shop_sum.columns=['user_id','shop_id','user_shop_sum']
        base=pd.merge(base,user_shop_sum,on=['user_id','shop_id'],how='left')
        base=base.fillna(0)

        '用户与商店的消费数量'
        user_shop=user_shop[user_shop['is_trade']==1]
        user_shop_trade=user_shop.groupby(by=['user_id','shop_id']).count().reset_index()
        user_shop_trade.columns=['user_id','shop_id','user_shop_trade']
        base=pd.merge(base,user_shop_trade,on=['user_id','shop_id'],how='left')
        base=base.fillna(0)


        # '商店对应的消费用户数及比例'
        # tmp=pd.concat([shop_id,user],axis=1)
        # tmp=tmp[tmp['is_trade']==1]
        # tmp=tmp.drop(['is_trade'],axis=1)
        # shop_user_trade_count=tmp.groupby(by='shop_id').count().reset_index()
        # shop_user_trade_count.columns=['shop_id','shop_user_trade_count']
        # base=pd.merge(base,shop_user_trade_count,on='shop_id',how='left')
        # base['shop_user_trade_ratio']=base['shop_user_trade_count']/base['shop_user_sum']
        # base=base.fillna(-1)

        # '商店对应的未消费的用户数及比例'
        # tmp=pd.concat([shop_id,user],axis=1)
        # tmp=tmp[tmp['is_trade']==0]
        # tmp=tmp.drop(['is_trade'],axis=1)
        # shop_user_untrade_count=tmp.groupby(by='shop_id').count().reset_index()
        # shop_user_untrade_count.columns=['shop_id','shop_user_untrade_count']
        # base=pd.merge(base,shop_user_untrade_count,on='shop_id',how='left')
        # base['shop_user_untrade_ratio']=base['shop_user_untrade_count']/base['shop_user_sum']
        # base=base.fillna(-1)

        # base=base.drop(['user_id','shop_id'],axis=1)
        # base=base.drop(['instance_id'],axis=1)

        return base
    def User_Context(self,f,base):
        '每小时的相关特征'

        hour_user=f[['context_timestamp','user_id','is_trade']]
        '每小时的用户量'
        def get_hour(x):
            return x.hour
        hour_user['context_timestamp'] = hour_user['context_timestamp'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        hour_user['hour'] = hour_user['context_timestamp'].map(get_hour)
        hour_user=hour_user.drop(['context_timestamp'],axis=1)

        # base['hour']=base['context_timestamp'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        # base['hour']=base['hour'].map(get_hour)
        # base=base.drop(['context_timestamp'],axis=1)

        time_user_sum=hour_user.groupby(by=['user_id','hour']).count().reset_index()
        time_user_sum.columns=['user_id','hour','time_user_sum']
        base=pd.merge(base,time_user_sum,on=['user_id','hour'],how='left')
        base=base.fillna(0)


        # '每小时用户消费量及比例'
        # user_trade = user[['user_id', 'is_trade']]
        # user_trade=pd.concat([context['time'],user_trade],axis=1)
        # user_trade=user_trade[user_trade['is_trade']==1]
        # user_trade=user_trade.drop(['is_trade'],axis=1)
        # time_user_count=user_trade.groupby(by='time').count().reset_index()
        # time_user_count.columns=['time','time_user_count']
        # base=pd.merge(base,time_user_count,on='time',how='left')
        # base['time_user_ratio']=base['time_user_count']/base['time_user_sum']



        return base
        pass
    def Ad_Context(self,f,base):
        hour_item=f[['context_timestamp','item_id','is_trade']]


        def get_hour(x):
            return x.hour
        hour_item['context_timestamp'] = hour_item['context_timestamp'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        hour_item['hour'] = hour_item['context_timestamp'].map(get_hour)
        hour_item=hour_item.drop(['context_timestamp'],axis=1)

        # base['hour']=base['context_timestamp'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        # base['hour']=base['hour'].map(get_hour)


        '广告小时的总量'
        hour_item_sum=hour_item.groupby(by=['item_id','hour']).count().reset_index()
        hour_item_sum.columns=['item_id','hour','hour_item_sum']
        base=pd.merge(base,hour_item_sum,on=['item_id','hour'],how='left')
        base=base.fillna(0)

        '广告小时的消费量'
        hour_item_count=hour_item[hour_item['is_trade']==1]
        hour_item_count=hour_item_count.groupby(by=['item_id','hour']).count().reset_index()
        hour_item_count.columns=['item_id','hour','hour_item_count']
        base=pd.merge(base,hour_item_count,on=['item_id','hour'],how='left')
        base=base.fillna(0)

        return base
        pass
    def Shop_Context(self,f,base):
        hour_shop=f[['context_timestamp','shop_id','is_trade']]

        def get_hour(x):
            return x.hour

        hour_shop['context_timestamp'] = hour_shop['context_timestamp'].map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        hour_shop['hour'] = hour_shop['context_timestamp'].map(get_hour)
        hour_shop = hour_shop.drop(['context_timestamp'], axis=1)

        # base['hour'] = base['context_timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        # base['hour'] = base['hour'].map(get_hour)
        # base = base.drop(['context_timestamp'], axis=1)

        '每小时的商店量'
        hour_shop_sum=hour_shop.groupby(by=['shop_id','hour']).count().reset_index()
        hour_shop_sum.columns=['shop_id','hour','hour_shop_sum']
        base=pd.merge(base,hour_shop_sum,on=['shop_id','hour'],how='left')
        base=base.fillna(0)

        '每小时的商店消费量'
        hour_shop_count=hour_shop[hour_shop['is_trade']==1]
        hour_shop_count=hour_shop_count.groupby(by=['shop_id','hour']).count().reset_index()
        hour_shop_count.columns=['shop_id','hour','hour_shop_count']
        base=pd.merge(base,hour_shop_count,on=['shop_id','hour'],how='left')
        base=base.fillna(0)

        return base
        pass
    def User_Shop_Context(self,f,base):
        user_shop_hour=f[['context_timestamp','user_id','shop_id','is_trade']]

        def get_hour(x):
            return x.hour

        user_shop_hour['context_timestamp'] = user_shop_hour['context_timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        user_shop_hour['hour'] = user_shop_hour['context_timestamp'].map(get_hour)
        user_shop_hour = user_shop_hour.drop(['context_timestamp'], axis=1)

        'user_shop与hour的统计次数'
        user_shop_hour_sum=user_shop_hour.groupby(by=['user_id','shop_id','hour']).count().reset_index()
        user_shop_hour_sum.columns=['user_id','shop_id','hour','user_shop_hour_sum']
        base=pd.merge(base,user_shop_hour_sum,on=['user_id','shop_id','hour'],how='left')
        base=base.fillna(0)

        'user_shop与hour的交易次数'
        user_shop_hour_count=user_shop_hour[user_shop_hour['is_trade']==1]
        user_shop_hour_count = user_shop_hour_count.groupby(by=['user_id', 'shop_id', 'hour']).count().reset_index()
        user_shop_hour_count.columns = ['user_id', 'shop_id', 'hour', 'user_shop_hour_count']
        base = pd.merge(base, user_shop_hour_count, on=['user_id', 'shop_id', 'hour'], how='left')
        base = base.fillna(0)

        return base
        pass
    def User_Ad_Context(self,f,base):
        user_item_hour=f[['user_id','item_id','context_timestamp','is_trade']]

        def get_hour(x):
            return x.hour

        user_item_hour['context_timestamp'] = user_item_hour['context_timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        user_item_hour['hour'] = user_item_hour['context_timestamp'].map(get_hour)
        user_item_hour = user_item_hour.drop(['context_timestamp'], axis=1)

        'user_item 与 hour的次数'
        user_item_hour_sum=user_item_hour.groupby(by=['user_id','item_id','hour']).count().reset_index()
        user_item_hour_sum.columns=['user_id','item_id','hour','user_item_hour_sum']
        base=pd.merge(base,user_item_hour_sum,on=['user_id','item_id','hour'],how='left')
        base=base.fillna(0)

        'user_item 与 hour的交易次数'
        user_item_hour_count=user_item_hour[user_item_hour['is_trade']==1]
        user_item_hour_count=user_item_hour_count.groupby(by=['user_id','item_id','hour']).count().reset_index()
        user_item_hour_count.columns=['user_id','item_id','hour','user_item_hour_count']
        base=pd.merge(base,user_item_hour_count,on=['user_id','item_id','hour'],how='left')
        base=base.fillna(0)

        return base

class Basic:
    def OneHot(self,label):
        pass
    def Sum(self,label):
        user=label[['user_id','instance_id']]
        grouped=pd.pivot_table(user,index=['user_id'],values=['instance_id'],aggfunc='count').reset_index()
        grouped.columns=['user_id','user_sum_label']
        label=pd.merge(label,grouped,on='user_id',how='left')

        item=label[['item_id','instance_id']]
        grouped=pd.pivot_table(item,index=['item_id'],values=['instance_id'],aggfunc='count').reset_index()
        grouped.columns=['item_id','item_sum_label']
        label=pd.merge(label,grouped,on='item_id',how='left')

        shop=label[['shop_id','instance_id']]
        grouped=pd.pivot_table(shop,index=['shop_id'],values=['instance_id'],aggfunc='count').reset_index()
        grouped.columns=['shop_id','shop_sum_label']
        label=pd.merge(label,grouped,on='shop_id',how='left')

        user_item=label[['user_id','item_id','instance_id']]
        grouped=pd.pivot_table(user_item,index=['user_id','item_id'],values=['instance_id'],aggfunc='count').reset_index()
        grouped.columns=['user_id','item_id','user_item_sum_label']
        label=pd.merge(label,grouped,on=['user_id','item_id'],how='left')

        user_shop=label[['user_id','shop_id','instance_id']]
        grouped=pd.pivot_table(user_shop,index=['user_id','shop_id'],values=['instance_id'],aggfunc='count').reset_index()
        grouped.columns=['user_id','shop_id','user_shop_sum_label']
        label=pd.merge(label,grouped,on=['user_id','shop_id'],how='left')

        label=label.fillna(0)

        label['context_timestamp']= label['context_timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

        group=label[['shop_id','context_timestamp']]
        group_rank=group.groupby(by=['shop_id']).rank(ascending=False)
        label['shop_last_rank']=group_rank
        label['shop_if_last']=label['shop_last_rank'].map(lambda x:1 if x==1 else 0)

        group = label[['shop_id', 'context_timestamp']]
        group_rank = group.groupby(by=['shop_id']).rank(ascending=True)
        label['shop_first_rank'] = group_rank
        label['shop_if_first'] = label['shop_first_rank'].map(lambda x: 1 if x == 1 else 0)

        group=label[['user_id','context_timestamp']]
        group_rank=group.groupby(by='user_id').rank(ascending=False)
        label['user_last_rank']=group_rank
        label['user_if_last'] = label['user_last_rank'].map(lambda x: 1 if x == 1 else 0)

        group = label[['user_id', 'context_timestamp']]
        group_rank = group.groupby(by='user_id').rank(ascending=True)
        label['user_first_rank'] = group_rank
        label['user_if_first'] = label['user_first_rank'].map(lambda x: 1 if x == 1 else 0)

        group=label[['item_id','context_timestamp']]
        group_rank=group.groupby(by=['item_id']).rank(ascending=False)
        label['item_last_rank']=group_rank
        label['item_if_last'] = label['item_last_rank'].map(lambda x: 1 if x == 1 else 0)

        group = label[['item_id', 'context_timestamp']]
        group_rank = group.groupby(by=['item_id']).rank(ascending=True)
        label['item_first_rank'] = group_rank
        label['item_if_first'] = label['item_first_rank'].map(lambda x: 1 if x == 1 else 0)

        group=label[['user_id','item_id','context_timestamp']]
        group_rank=group.groupby(by=['user_id','item_id']).rank(ascending=False)
        label['user_item_last_rank']=group_rank
        label['user_item_if_last'] = label['user_item_last_rank'].map(lambda x: 1 if x == 1 else 0)

        group = label[['user_id', 'item_id', 'context_timestamp']]
        group_rank = group.groupby(by=['user_id', 'item_id']).rank(ascending=True)
        label['user_item_first_rank'] = group_rank
        label['user_item_if_first'] = label['user_item_first_rank'].map(lambda x: 1 if x == 1 else 0)

        group=label[['user_id','shop_id','context_timestamp']]
        group_rank=group.groupby(by=['user_id','shop_id']).rank(ascending=False)
        label['user_shop_last_rank']=group_rank
        label['user_shop_if_last'] = label['user_shop_last_rank'].map(lambda x: 1 if x == 1 else 0)

        group = label[['user_id', 'shop_id', 'context_timestamp']]
        group_rank = group.groupby(by=['user_id', 'shop_id']).rank(ascending=True)
        label['user_shop_first_rank'] = group_rank
        label['user_shop_if_first'] = label['user_shop_first_rank'].map(lambda x: 1 if x == 1 else 0)

        label['item_category'] = label['item_category_list'].map(lambda x: x.split(";"))
        label['item_category']=label['item_category'].map(lambda x:x[1] if len(x)<=2 else x[2])
        label['predict_category'] = label['predict_category_property'].map(
            lambda x: [y.split(':')[0] for y in x.split(';')])
        label['predict_category_rank'] = list(
            map(lambda x, y: y.index(x) if x in y else -1, label['item_category'],label['predict_category']))

        # l=['509660095530134768','5755694407684602296','5799347067982556520','7258015885215914736','8277336076276184272']
        # label['item_cat_dummies']=label['item_category'].map(lambda x:1 if x not in l else 0)
        # label['509660095530134768']=label['item_category'].map(lambda x:1 if x=='509660095530134768' else 0)
        # label['5755694407684602296'] = label['item_category'].map(lambda x: 1 if x == '5755694407684602296' else 0)
        # label['5799347067982556520'] = label['item_category'].map(lambda x: 1 if x == '5799347067982556520' else 0)
        # label['7258015885215914736'] = label['item_category'].map(lambda x: 1 if x == '7258015885215914736' else 0)
        # label['8277336076276184272'] = label['item_category'].map(lambda x: 1 if x == '8277336076276184272' else 0)

        tmp=label[['item_category','instance_id']]
        grouped=pd.pivot_table(tmp,index=['item_category'],values=['instance_id'],aggfunc='count').reset_index()
        grouped.columns=['item_category','item_category_sum']
        Max = grouped['item_category_sum'].max()
        Min = grouped['item_category_sum'].min()
        M = Max - Min
        grouped['item_category_sum'] = grouped['item_category_sum'].map(lambda x: (x - Min) / M)
        label=pd.merge(label,grouped,on='item_category',how='left')

        label['item_property_list'] = label['item_property_list'].map(lambda x: x.split(";"))
        label['property_len'] = label['item_property_list'].map(lambda x: len(x))

        label['item_pro1']=label['item_property_list'].map(lambda x:x[0])
        tmp=label[['item_pro1','instance_id']]
        grouped=pd.pivot_table(tmp,index='item_pro1',values='instance_id',aggfunc='count').reset_index()
        grouped.columns=['item_pro1','item_pro1_sum']
        label=pd.merge(label,grouped,on='item_pro1',how='left')

        label['item_pro2'] = label['item_property_list'].map(lambda x: x[1])
        tmp = label[['item_pro2', 'instance_id']]
        grouped = pd.pivot_table(tmp, index='item_pro2', values='instance_id', aggfunc='count').reset_index()
        grouped.columns = ['item_pro2', 'item_pro2_sum']
        label = pd.merge(label, grouped, on='item_pro2', how='left')

        label['item_pro3'] = label['item_property_list'].map(lambda x: x[2])
        tmp = label[['item_pro3', 'instance_id']]
        grouped = pd.pivot_table(tmp, index='item_pro3', values='instance_id', aggfunc='count').reset_index()
        grouped.columns = ['item_pro3', 'item_pro3_sum']
        label = pd.merge(label, grouped, on='item_pro3', how='left')

        label['item_pro4'] = label['item_property_list'].map(lambda x: x[3])
        tmp = label[['item_pro4', 'instance_id']]
        grouped = pd.pivot_table(tmp, index='item_pro4', values='instance_id', aggfunc='count').reset_index()
        grouped.columns = ['item_pro4', 'item_pro4_sum']
        label = pd.merge(label, grouped, on='item_pro4', how='left')

        tmp=label[['user_id','item_pro1','instance_id']]
        grouped=pd.pivot_table(tmp,index=['user_id','item_pro1'],values=['instance_id'],aggfunc='count').reset_index()
        grouped.columns=['user_id','item_pro1','user_pro1_sum']
        label=pd.merge(label,grouped,on=['user_id','item_pro1'],how='left')

        tmp = label[['item_id', 'item_pro1', 'instance_id']]
        grouped = pd.pivot_table(tmp, index=['item_id', 'item_pro1'], values=['instance_id'],\
                                 aggfunc='count').reset_index()
        grouped.columns = ['item_id', 'item_pro1', 'item_pro1_sum']
        label = pd.merge(label, grouped, on=['item_id', 'item_pro1'], how='left')

        tmp = label[['shop_id', 'item_pro1', 'instance_id']]
        grouped = pd.pivot_table(tmp, index=['shop_id', 'item_pro1'], values=['instance_id'], \
                                 aggfunc='count').reset_index()
        grouped.columns = ['shop_id', 'item_pro1', 'shop_pro1_sum']
        label = pd.merge(label, grouped, on=['shop_id', 'item_pro1'], how='left')

        tmp = label[['user_id','shop_id','item_pro1', 'instance_id']]
        grouped = pd.pivot_table(tmp, index=['user_id','shop_id', 'item_pro1'], values=['instance_id'], \
                                 aggfunc='count').reset_index()
        grouped.columns = ['user_id','shop_id', 'item_pro1', 'user_shop_pro1_sum']
        label = pd.merge(label, grouped, on=['user_id','shop_id', 'item_pro1'], how='left')

        tmp = label[['user_id', 'item_id', 'item_pro1', 'instance_id']]
        grouped = pd.pivot_table(tmp, index=['user_id', 'item_id', 'item_pro1'], values=['instance_id'], \
                                 aggfunc='count').reset_index()
        grouped.columns = ['user_id', 'item_id', 'item_pro1', 'user_item_pro1_sum']
        label = pd.merge(label, grouped, on=['user_id', 'item_id', 'item_pro1'], how='left')

        tmp=label[['item_pro1','context_timestamp']]
        group_rank=tmp.groupby(by='item_pro1').rank(ascending=False)
        label['item_pro1_last_rank']=group_rank
        label['item_pro1_if_last']=label['item_pro1_last_rank'].map(lambda x:1 if x==1 else 0)

        tmp = label[['item_pro1', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro1').rank(ascending=True)
        label['item_pro1_first_rank'] = group_rank
        label['item_pro1_if_first'] = label['item_pro1_first_rank'].map(lambda x: 1 if x == 1 else 0)

        tmp = label[['item_pro2', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro2').rank(ascending=True)
        label['item_pro2_first_rank'] = group_rank
        label['item_pro2_if_first'] = label['item_pro2_first_rank'].map(lambda x: 1 if x == 1 else 0)

        tmp = label[['item_pro2', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro2').rank(ascending=False)
        label['item_pro2_last_rank'] = group_rank
        label['item_pro2_if_last'] = label['item_pro2_last_rank'].map(lambda x: 1 if x == 1 else 0)

        tmp = label[['item_pro3', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro3').rank(ascending=True)
        label['item_pro3_first_rank'] = group_rank
        label['item_pro3_if_first'] = label['item_pro3_first_rank'].map(lambda x: 1 if x == 1 else 0)

        tmp = label[['item_pro3', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro3').rank(ascending=False)
        label['item_pro3_last_rank'] = group_rank
        label['item_pro3_if_last'] = label['item_pro3_last_rank'].map(lambda x: 1 if x == 1 else 0)

        tmp = label[['item_pro4', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro4').rank(ascending=False)
        label['item_pro4_last_rank'] = group_rank
        label['item_pro4_if_last'] = label['item_pro4_last_rank'].map(lambda x: 1 if x == 1 else 0)

        tmp = label[['item_pro4', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro4').rank(ascending=True)
        label['item_pro4_first_rank'] = group_rank
        label['item_pro4_if_first'] = label['item_pro4_first_rank'].map(lambda x: 1 if x == 1 else 0)

        label['item_pro5']=label['item_property_list'].map(lambda x:x[4])
        tmp=label[['item_pro5','context_timestamp']]
        group_rank=tmp.groupby(by='item_pro5').rank(ascending=True)
        label['item_pro5_first_rank']=group_rank
        label['item_pro5_if_first']=label['item_pro5_first_rank'].map(lambda x:1 if x==1 else 0)

        tmp = label[['item_pro5', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro5').rank(ascending=False)
        label['item_pro5_last_rank'] = group_rank
        label['item_pro5_if_last'] = label['item_pro5_last_rank'].map(lambda x: 1 if x == 1 else 0)

        label['item_pro6']=label['item_property_list'].map(lambda x:x[5])
        tmp=label[['item_pro6','context_timestamp']]
        group_rank=tmp.groupby(by='item_pro6').rank(ascending=True)
        label['item_pro6_first_rank']=group_rank
        label['item_pro6_if_first']=label['item_pro6_first_rank'].map(lambda x:1 if x==1 else 0)

        tmp = label[['item_pro6', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro6').rank(ascending=False)
        label['item_pro6_last_rank'] = group_rank
        label['item_pro6_if_last'] = label['item_pro6_last_rank'].map(lambda x: 1 if x == 1 else 0)

        label['item_pro7'] = label['item_property_list'].map(lambda x: x[6])
        tmp = label[['item_pro7', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro7').rank(ascending=True)
        label['item_pro7_first_rank'] = group_rank
        label['item_pro7_if_first'] = label['item_pro7_first_rank'].map(lambda x: 1 if x == 1 else 0)

        tmp = label[['item_pro7', 'context_timestamp']]
        group_rank = tmp.groupby(by='item_pro7').rank(ascending=False)
        label['item_pro7_last_rank'] = group_rank
        label['item_pro7_if_last'] = label['item_pro7_last_rank'].map(lambda x: 1 if x == 1 else 0)


        label=label.drop(['item_category','predict_category','predict_category_property',\
                          'item_property_list','item_pro1','item_pro2','item_pro3','item_pro4',\
                          'item_pro5','item_pro6','item_pro7'],axis=1)

        tmp=label[['user_id','predict_category_rank','instance_id']]
        grouped=pd.pivot_table(tmp,index=['user_id','predict_category_rank'],\
                               values=['instance_id'],aggfunc='count',fill_value=0).reset_index()
        grouped.columns=['user_id','predict_category_rank','user_predict_sum']
        label=pd.merge(label,grouped,on=['user_id','predict_category_rank'],how='left')

        tmp=label[['shop_id','predict_category_rank','instance_id']]
        grouped=pd.pivot_table(tmp,index=['shop_id','predict_category_rank'],\
                               values=['instance_id'],aggfunc='count',fill_value=0).reset_index()
        grouped.columns=['shop_id','predict_category_rank','shop_preidct_sum']
        label=pd.merge(label,grouped,on=['shop_id','predict_category_rank'],how='left')

        tmp=label[['item_id','predict_category_rank','instance_id']]
        grouped=pd.pivot_table(tmp,index=['item_id','predict_category_rank'],\
                               values=['instance_id'],aggfunc='count',fill_value=0).reset_index()
        grouped.columns=['item_id','predict_category_rank','item_predict_sum']
        label=pd.merge(label,grouped,on=['item_id','predict_category_rank'],how='left')

        tmp=label[['user_id','item_id','predict_category_rank','instance_id']]
        grouped=pd.pivot_table(tmp,index=['user_id','item_id','predict_category_rank'],\
                               values=['instance_id'],fill_value=0).reset_index()
        grouped.columns=['user_id','item_id','predict_category_rank','user_item_pre_sum']
        label=pd.merge(label,grouped,on=['user_id','item_id','predict_category_rank'],how='left')

        tmp=label[['user_id','shop_id','predict_category_rank','instance_id']]
        grouped=pd.pivot_table(tmp,index=['user_id','shop_id','predict_category_rank'],\
                               values=['instance_id'],aggfunc='count',fill_value=0).reset_index()
        grouped.columns=['user_id','shop_id','predict_category_rank','user_shop_pre_sum']
        label=pd.merge(label,grouped,on=['user_id','shop_id','predict_category_rank'],how='left')

        return label
def addlabel(label):
    label['item_property_list'] = label['item_property_list'].map(lambda x: x.split(";"))
    label['item_pro2'] = label['item_property_list'].map(lambda x: x[1])
    tmp=label['item_pro2','instance_id']
    grouped = pd.pivot_table(tmp, index='item_pro2', values='instance_id', aggfunc='count').reset_index()
    grouped.columns = ['item_pro2', 'item_pro2_sum']
    label = pd.merge(label, grouped, on='item_pro2', how='left')

    label['item_pro3'] = label['item_property_list'].map(lambda x: x[2])
    tmp = label['item_pro3', 'instance_id']
    grouped = pd.pivot_table(tmp, index='item_pro3', values='instance_id', aggfunc='count').reset_index()
    grouped.columns = ['item_pro3', 'item_pro3_sum']
    label = pd.merge(label, grouped, on='item_pro3', how='left')

    return label
    pass
def ADD():
    # train_laset=pd.read_csv(r'G:\Tianchi\IJCAI-18\Featforparams\train_laset.csv')
    # train_label=pd.read_csv(r'')
    pass
if __name__ == '__main__':

    pass