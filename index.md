---
layout: home
title: "Survival Analysis"
date: 2024-04-12
---

## 1.Introduction

### 生存分析简介
生存分析是一种统计方法，主要用于研究生存时间（从起点事件到终点事件的时间）的分布规律以及影响生存时间的因素。它广泛应用于医学、工程、社会科学等领域，能够处理数据中的删失现象（即部分数据未完全观察到终点事件），通过构建生存函数和风险函数等模型，帮助研究者分析和预测生存时间的特征。

**应用例子**：
1. **医学领域**：在临床试验中，研究某种新药对癌症患者生存时间的影响，通过生存分析比较用药组和对照组的生存曲线，评估药物的疗效。
2. **工程领域**：对机械设备的使用寿命进行分析，通过生存分析预测设备的故障时间分布，从而制定合理的维护和更换计划。
3. **社会科学领域**：研究失业人员再就业的时间分布，分析不同因素（如年龄、教育水平、技能等）对再就业时间的影响，为制定就业政策提供依据。

### 本笔记概要
本笔记将运用并回顾几种常用的生存分析技术：
1. Kaplan-Meier 估计法与对数秩检验
2. Cox 比例风险模型
3. 加速失效时间模型

最后，笔记将把生存分析模型的输出结果作为客户终身价值（Customer Lifetime Value）仪表板的输入数据。

在这一系列的分析中，我们使用的数据集来自 [IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)，它模拟了一家虚构的电信公司的运营情况。数据集中的每条记录代表一个订阅用户，涵盖了用户的人口统计信息、服务计划、媒体使用情况以及订阅状态等多方面的细节。其中包含了进行生存分析所需的两个关键字段：

**Tenure（在网时长）**：表示客户在流失之前的订阅时长，如果客户仍然是订阅者，则为当前的订阅时长。
**Churn（流失状态）**：一个布尔值，用于标记客户是否仍然是公司的订阅用户。1表示流失，0表示未流失。

在正式开始分析之前，我们需要依次完成以下步骤：

1. 下载 [IBM 的 Telco 数据集](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)。
2. 将数据存储到 Bronze 和 Silver 表中。


```python
from pyspark.sql import SparkSession

# 初始化 Spark 会话
spark = SparkSession.builder.appName("SurvivalAnalysis").getOrCreate()

from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import warnings
warnings.filterwarnings('ignore')

# 定义数据的字段的名称和数据类型
schema = StructType([
    StructField('customerID', StringType()),
    StructField('gender', StringType()),
    StructField('seniorCitizen', DoubleType()),
    StructField('partner', StringType()),
    StructField('dependents', StringType()),
    StructField('tenure', DoubleType()),
    StructField('phoneService', StringType()),
    StructField('multipleLines', StringType()),
    StructField('internetService', StringType()), 
    StructField('onlineSecurity', StringType()),
    StructField('onlineBackup', StringType()),
    StructField('deviceProtection', StringType()),
    StructField('techSupport', StringType()),
    StructField('streamingTV', StringType()),
    StructField('streamingMovies', StringType()),
    StructField('contract', StringType()),
    StructField('paperlessBilling', StringType()),
    StructField('paymentMethod', StringType()),
    StructField('monthlyCharges', DoubleType()),
    StructField('totalCharges', DoubleType()),
    StructField('churnString', StringType())
])

# 读取数据
bronze_df = spark.read.format('csv').schema(schema).option('header', 'true') \
            .load("Telco-Customer-Churn.csv")
```


```python
from pyspark.sql.functions import col, when

# 构造银级表
# 将 churn 列转换为布尔值；筛选出具有月度合同的 Internet 订阅者
silver_df = bronze_df.withColumn('churn', when(col('churnString') == 'Yes', 1) \
                                 .when(col('churnString') == 'No', 0).otherwise('Unknown')) \
                     .drop('churnString') \
                     .filter(col('contract') == 'Month-to-month') \
                     .filter(col('internetService') != 'No')
```


```python
# 创建临时视图（如果要看数据的话）
bronze_df.createOrReplaceTempView("bronze_customers")
silver_df.createOrReplaceTempView("silver_monthly_customers")

# 查询数据
result = spark.sql("""
SELECT * FROM silver_monthly_customers
""")
result.show(5)
```

    +----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+--------------------+--------------+------------+-----+
    |customerID|gender|seniorCitizen|partner|dependents|tenure|phoneService|   multipleLines|internetService|onlineSecurity|onlineBackup|deviceProtection|techSupport|streamingTV|streamingMovies|      contract|paperlessBilling|       paymentMethod|monthlyCharges|totalCharges|churn|
    +----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+--------------------+--------------+------------+-----+
    |7590-VHVEG|Female|          0.0|    Yes|        No|   1.0|          No|No phone service|            DSL|            No|         Yes|              No|         No|         No|             No|Month-to-month|             Yes|    Electronic check|         29.85|       29.85|    0|
    |3668-QPYBK|  Male|          0.0|     No|        No|   2.0|         Yes|              No|            DSL|           Yes|         Yes|              No|         No|         No|             No|Month-to-month|             Yes|        Mailed check|         53.85|      108.15|    1|
    |9237-HQITU|Female|          0.0|     No|        No|   2.0|         Yes|              No|    Fiber optic|            No|          No|              No|         No|         No|             No|Month-to-month|             Yes|    Electronic check|          70.7|      151.65|    1|
    |9305-CDSKC|Female|          0.0|     No|        No|   8.0|         Yes|             Yes|    Fiber optic|            No|          No|             Yes|         No|        Yes|            Yes|Month-to-month|             Yes|    Electronic check|         99.65|       820.5|    1|
    |1452-KIOVK|  Male|          0.0|     No|       Yes|  22.0|         Yes|             Yes|    Fiber optic|            No|         Yes|              No|         No|        Yes|             No|Month-to-month|             Yes|Credit card (auto...|          89.1|      1949.4|    0|
    +----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+--------------------+--------------+------------+-----+
    only showing top 5 rows
    


    25/04/10 12:43:46 WARN CSVHeaderChecker: CSV header does not conform to the schema.
     Header: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
     Schema: customerID, gender, seniorCitizen, partner, dependents, tenure, phoneService, multipleLines, internetService, onlineSecurity, onlineBackup, deviceProtection, techSupport, streamingTV, streamingMovies, contract, paperlessBilling, paymentMethod, monthlyCharges, totalCharges, churnString
    Expected: churnString but found: Churn
    CSV file: file:///data/lab/project1/survival-analysis/Telco-Customer-Churn.csv


## 2.Kaolan-Meier

### 目标
1. 将 Kaplan-Meier 生存概率曲线拟合到 IBM 的 Telco 数据集。
2. 直观地评估总体水平和协变量水平的生存概率曲线。
3. 使用对数秩检验来确定生存曲线在统计上是否等效。
4. 提取生存概率以供后续建模。

### Kaplan-Meier估计
Kaplan-Meier 估计（简称 K-M 估计）是一种非参数方法，用于估计生存函数，即在特定时间点仍存活的概率。其主要步骤包括：
1. 将数据按生存时间从小到大排序，删失数据排在相同时间点的非删失数据之后。
2. 从最小的时间点开始，逐步向右移动，每次遇到删失数据时，将其质量均匀重新分配给右侧的所有数据点。
3. 通过乘积极限法计算生存率，即在每个事件发生的时间点，生存率会根据事件发生的比例进行调整。

Kaplan-Meier 曲线以生存时间为横轴，生存率 $S(t)$ 为纵轴，绘制出阶梯状的曲线，直观地展示了生存时间与生存率之间的关系。


```python
# 由于 PySpark 本身不直接支持 Lifelines 库（这是一个 Python 的生存分析库），需要将 PySpark DataFrame 转换为 
# Pandas DataFrame，然后使用 Lifelines 进行分析。
telco_pd = spark.table('silver_monthly_customers').toPandas()
```

    25/04/11 13:53:46 WARN CSVHeaderChecker: CSV header does not conform to the schema.
     Header: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
     Schema: customerID, gender, seniorCitizen, partner, dependents, tenure, phoneService, multipleLines, internetService, onlineSecurity, onlineBackup, deviceProtection, techSupport, streamingTV, streamingMovies, contract, paperlessBilling, paymentMethod, monthlyCharges, totalCharges, churnString
    Expected: churnString but found: Churn
    CSV file: file:///data/lab/project1/survival-analysis/Telco-Customer-Churn.csv



```python
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import pairwise_logrank_test

# 初始化 Kaplan-Meier Fitter，这是一个类，用于拟合生存数据
kmf = KaplanMeierFitter()

# 提取生存时间和生存状态
T = telco_pd['tenure']
C = telco_pd['churn'].astype(float) # 1为流失，0为没流失

# 拟合模型
kmf.fit(T, C) # 拟合生存曲线
kmf.plot(title='Kaplan-Meier Survival Curve: Population level') # 画图
plt.show()

# 输出中位生存时间
print(f"Median Survival Time: {kmf.median_survival_time_}")

# 绘制按某个特征分组的生存曲线
def plot_km(col):
    ax = plt.subplot(111)
    for r in telco_pd[col].unique():
        ix = telco_pd[col] == r
        kmf.fit(T[ix], C[ix], label=r)
        kmf.plot(ax=ax)
    plt.title(f'Kaplan-Meier Survival Curves by {col}')
    plt.show()
```


    
![png](/image/output_6_0.png)
    


    Median Survival Time: 34.0


从最纯粹的意义上讲，客户至少留存 0 个月的概率是 100%。这由上图中的点 （0,1.0） 表示。将生存曲线向下移动到中位数（34 个月），可以说客户至少有 50% 的概率留存至少 34 个月，前提是他们已经留存了 33 个月。\
生存概率曲线周围的浅蓝色边框表示置信区间。区间越宽，置信度越低。如上图所示，估计值的置信度随着时间线的增加而降低。虽然这种置信度降低可能是由于数据较少，但同样可以理解为，我们对近期预测的信心比对长期预测的信心更大。

### Log-rank

对数秩检验（log-rank test）是一种用于比较两组或多组生存曲线是否存在显著差异的非参数检验方法。其基本原理包括：
1. 将所有组的生存时间混合后按从小到大排序，形成多个时间点。
2. 在每个时间点，计算实际发生的事件数（如死亡）和理论预期的事件数。
3. 检验统计量为卡方统计量，其计算公式为：
   $$
   \chi^2 = \sum \frac{(A - T)^2}{T}
   $$
   其中 $A$ 为实际事件数，$ T $ 为理论事件数。
4. 如果计算得到的 $ p $ 值小于设定的显著性水平（如 0.05），则拒绝零假设，认为两组生存曲线存在显著差异。

需要注意的是，对数秩检验假设各组的风险比例随时间保持不变，如果生存曲线存在交叉，则可能违反这一假设，导致检验结果不准确。


```python
# 绘制按某个特征分组的生存曲线
def plot_km(col):
    ax = plt.subplot(111)
    for r in telco_pd[col].unique():
        ix = telco_pd[col] == r
        kmf.fit(T[ix], C[ix], label=r)
        kmf.plot(ax=ax)
    plt.title(f'Kaplan-Meier Survival Curves by {col}')
    plt.show()

# 输出 Log-rank 检验结果，用于比较不同组之间的生存差异
def print_logrank(col):
    log_rank = pairwise_logrank_test(telco_pd['tenure'], telco_pd[col], telco_pd['churn'])
    print(log_rank.summary)

# 对性别(gender)绘制生存曲线并输出 Log-rank 检验结果
plot_km('gender')
print_logrank('gender')
```


    
![png](/image/output_9_0.png)
    


                 test_statistic         p  -log2(p)
    Female Male         1.61011  0.204476  2.289995


对数秩检验中的 p 值大于 0.05，因此，我们不能拒绝两组在统计上相等的原假设。这意味着在当前的显著性水平下，没有足够的证据表明客户的性别对留存时间有显著影响。


```python
# 还可以对更多特征绘制生存曲线并输出 Log-rank 检验结果
plot_km('onlineSecurity')
print_logrank('onlineSecurity')

plot_km('seniorCitizen')
print_logrank('seniorCitizen')

plot_km('partner')
print_logrank('partner')

plot_km('dependents')
print_logrank('dependents')

plot_km('phoneService')
print_logrank('phoneService')

plot_km('multipleLines')
print_logrank('multipleLines')

plot_km('internetService')
print_logrank('internetService')

plot_km('streamingTV')
print_logrank('streamingTV')

plot_km('streamingMovies')
print_logrank('streamingMovies')

plot_km('onlineBackup')
print_logrank('onlineBackup')

plot_km('deviceProtection')
print_logrank('deviceProtection')

plot_km('techSupport')
print_logrank('techSupport')

plot_km('paperlessBilling')
print_logrank('paperlessBilling')

plot_km('paymentMethod')
print_logrank('paymentMethod')
```


    
![png](/image/output_11_0.png)
    


            test_statistic             p   -log2(p)
    No Yes       75.800079  3.138886e-18  58.144453



    
![png](/image/output_11_2.png)
    


             test_statistic             p  -log2(p)
    0.0 1.0       49.027784  2.523624e-12  38.52764



    
![png](/image/output_11_4.png)
    


            test_statistic             p    -log2(p)
    No Yes      257.844159  5.063437e-58  190.331712



    
![png](/image/output_11_6.png)
    


            test_statistic         p  -log2(p)
    No Yes       13.405914  0.000251  11.96099



    
![png](/image/output_11_8.png)
    


            test_statistic         p  -log2(p)
    No Yes        0.778505  0.377599  1.405074



    
![png](/image/output_11_10.png)
    


                                       test_statistic             p    -log2(p)
    No               No phone service       35.546250  2.490661e-09   28.580824
                     Yes                   411.225536  1.983168e-91  301.307649
    No phone service Yes                    44.056629  3.190116e-11   34.867600



    
![png](/image/output_11_12.png)
    


                     test_statistic             p   -log2(p)
    DSL Fiber optic       85.455399  2.369872e-20  65.193753



    
![png](/image/output_11_14.png)
    


            test_statistic             p    -log2(p)
    No Yes      140.761789  1.813974e-32  105.442545



    
![png](/image/output_11_16.png)
    


            test_statistic             p    -log2(p)
    No Yes      170.262183  6.484901e-39  126.858111



    
![png](/image/output_11_18.png)
    


            test_statistic             p    -log2(p)
    No Yes      300.455875  2.620909e-67  221.179115



    
![png](/image/output_11_20.png)
    


            test_statistic             p    -log2(p)
    No Yes      169.868512  7.904692e-39  126.572486



    
![png](/image/output_11_22.png)
    


            test_statistic             p   -log2(p)
    No Yes       25.969416  3.468692e-07  21.459105



    
![png](/image/output_11_24.png)
    


            test_statistic             p   -log2(p)
    No Yes       25.263459  5.000937e-07  20.931298



    
![png](/image/output_11_26.png)
    


                                                       test_statistic  \
    Bank transfer (automatic) Credit card (automatic)        0.153545   
                              Electronic check              55.164654   
                              Mailed check                 190.000457   
    Credit card (automatic)   Electronic check              45.167592   
                              Mailed check                 165.361074   
    Electronic check          Mailed check                  72.323100   
    
                                                                  p    -log2(p)  
    Bank transfer (automatic) Credit card (automatic)  6.951703e-01    0.524562  
                              Electronic check         1.108442e-13   43.036532  
                              Mailed check             3.178566e-43  141.174532  
    Credit card (automatic)   Electronic check         1.808736e-11   35.686227  
                              Mailed check             7.628420e-38  123.301883  
    Electronic check          Mailed check             1.826962e-17   55.603331  


## 3.Cox Proportional Hazards

### 目标
在本课程中，您将完成以下任务：

1. 将 Cox 比例风险模型拟合到 IBM 的 Telco 数据集
2. 解释 Cox 比例风险模型的统计输出
3. 确定模型是否遵循比例风险假设

### Cox 比例风险
Kaplan-Meier：是一种非参数方法，用于估计生存函数（即随时间变化的生存概率）。它不假设数据的分布形式，适用于描述性分析。主要用于生成生存曲线，直观展示不同组之间的生存差异，适合初步探索和描述性分析。\
Cox 比例风险模型：是一种半参数模型，用于分析影响生存时间的因素。它假设风险比是与时间无关的（比例风险假设），但不假设基线风险函数的具体形式。用于评估多个协变量对生存时间的影响，适合进行因果推断和预测分析。

Kaplan-Meier 方法主要用于描述性分析，通过生存曲线直观地展示不同组（如不同治疗组、不同性别等）的生存差异。\
Cox 模型用于评估多个协变量（如年龄、性别、治疗方式等）对生存时间的影响，确定哪些变量对生存时间有更显著影响。比如Cox 模型会给出每个协变量的风险比（Hazard Ratio, HR），表示该变量对生存时间的影响大小。风险比大于1表示该变量增加风险，小于1表示降低风险。

Cox 风险比是基线风险和部分风险的乘积，基线风险是当所有协变量 X 都为 0（默认值） 时的风险，部分风险表示当协变量的值与基线不同时发生的风险变化。它是协变量的线性组合的指数函数。

### 比例风险假设
在Cox比例风险模型中，基线风险是时间 $t$ 的函数，而与模型参数无关；相对地，部分风险（即与协变量相关的风险）是参数的函数，而与时间无关。

在Cox比例风险模型的框架下，任意两组之间的风险比在不同时间点上保持恒定，即风险比与时间无关。因为部分风险不随时间 $t$ 变化，这意味着当某个因素改变时，它对风险的影响是恒定的，不会随时间而改变。

### One-Hot编码
使用 Pandas 的 get_dummies() 函数对分类变量进行 One-Hot 编码。为了防止多重共线性问题，需要在编码后删除每一组中的一个冗余列。例如，对于变量 dependents，编码后会生成 dependents_yes 和 dependents_no 两列，您需要删除其中一列（通常删除 dependents_no）。

另一种方法是手动选择要删除的变量。在本笔本中，我们选择了删除那些与总体 Kaplan-Meier 生存概率曲线最相似的值对应的列。


```python
import pandas as pd

# 编码分类变量，one-hot 编码，将每个分类变量的每个类别转换为一个二进制列（0 或 1），表示该类别是否存在
encode_cols = ['dependents', 'internetService', 'onlineBackup', 'techSupport', 'paperlessBilling']
encoded_pd = pd.get_dummies(telco_pd, columns=encode_cols, prefix=encode_cols, drop_first=False)
encoded_pd.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>seniorCitizen</th>
      <th>partner</th>
      <th>tenure</th>
      <th>phoneService</th>
      <th>multipleLines</th>
      <th>onlineSecurity</th>
      <th>deviceProtection</th>
      <th>streamingTV</th>
      <th>...</th>
      <th>dependents_No</th>
      <th>dependents_Yes</th>
      <th>internetService_DSL</th>
      <th>internetService_Fiber optic</th>
      <th>onlineBackup_No</th>
      <th>onlineBackup_Yes</th>
      <th>techSupport_No</th>
      <th>techSupport_Yes</th>
      <th>paperlessBilling_No</th>
      <th>paperlessBilling_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>1.0</td>
      <td>No</td>
      <td>No phone service</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0.0</td>
      <td>No</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0.0</td>
      <td>No</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9305-CDSKC</td>
      <td>Female</td>
      <td>0.0</td>
      <td>No</td>
      <td>8.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1452-KIOVK</td>
      <td>Male</td>
      <td>0.0</td>
      <td>No</td>
      <td>22.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
survival_pd = encoded_pd[['churn', 'tenure', 'dependents_Yes', 'internetService_DSL', 'onlineBackup_Yes', 
                          'techSupport_Yes']]

# 将 'churn' 列转换为浮点类型， Lifelines要求的
survival_pd['churn'] = survival_pd['churn'].astype('float')

from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# 初始化 Cox 比例风险模型，95% 的置信区间
cph = CoxPHFitter(alpha=0.05)
cph.fit(survival_pd, 'tenure', 'churn')

# 打印模型摘要
cph.print_summary()
```

    /tmp/ipykernel_16780/1555761253.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      survival_pd['churn'] = survival_pd['churn'].astype('float')



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'tenure'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'churn'</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>3351</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>1556</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-11315.95</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2025-04-10 14:56:40 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dependents_Yes</th>
      <td>-0.33</td>
      <td>0.72</td>
      <td>0.07</td>
      <td>-0.47</td>
      <td>-0.19</td>
      <td>0.63</td>
      <td>0.83</td>
      <td>0.00</td>
      <td>-4.64</td>
      <td>&lt;0.005</td>
      <td>18.12</td>
    </tr>
    <tr>
      <th>internetService_DSL</th>
      <td>-0.22</td>
      <td>0.80</td>
      <td>0.06</td>
      <td>-0.33</td>
      <td>-0.10</td>
      <td>0.72</td>
      <td>0.90</td>
      <td>0.00</td>
      <td>-3.68</td>
      <td>&lt;0.005</td>
      <td>12.07</td>
    </tr>
    <tr>
      <th>onlineBackup_Yes</th>
      <td>-0.78</td>
      <td>0.46</td>
      <td>0.06</td>
      <td>-0.89</td>
      <td>-0.66</td>
      <td>0.41</td>
      <td>0.52</td>
      <td>0.00</td>
      <td>-13.13</td>
      <td>&lt;0.005</td>
      <td>128.37</td>
    </tr>
    <tr>
      <th>techSupport_Yes</th>
      <td>-0.64</td>
      <td>0.53</td>
      <td>0.08</td>
      <td>-0.79</td>
      <td>-0.49</td>
      <td>0.46</td>
      <td>0.61</td>
      <td>0.00</td>
      <td>-8.48</td>
      <td>&lt;0.005</td>
      <td>55.36</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.64</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>22639.90</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>337.77 on 4 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>236.24</td>
    </tr>
  </tbody>
</table>
</div>


每列的 p 值低于 < 0.05。因此，每列都具有统计显著性

第1列是系数。以 internetService_DSL 为例，当客户为其互联网服务订阅 DSL 时，其风险率会相对于基线降低一个系数，也就是基线的0.8倍


```python
# 绘制风险比图，看置信区间
cph.plot(hazard_ratios=True)
plt.show()

# 检查比例风险假设
cph.check_assumptions(survival_pd, p_value_threshold=0.05, show_plots=True)
```


    
![png](/image/output_16_0.png)
    


    
       Bootstrapping lowess lines. May take a moment...
    
    
       Bootstrapping lowess lines. May take a moment...
    
    The ``p_value_threshold`` is set at 0.05. Even under the null hypothesis of no violations, some
    covariates will be below the threshold by chance. This is compounded when there are many covariates.
    Similarly, when there are lots of observations, even minor deviances from the proportional hazard
    assumption will be flagged.
    
    With that in mind, it's best to use a combination of statistical tests and visual tests to determine
    the most serious violations. Produce visual plots using ``check_assumptions(..., show_plots=True)``
    and looking for non-constant lines. See link [A] below for a full example.
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>null_distribution</th>
      <td>chi squared</td>
    </tr>
    <tr>
      <th>degrees_of_freedom</th>
      <td>1</td>
    </tr>
    <tr>
      <th>model</th>
      <td>&lt;lifelines.CoxPHFitter: fitted with 3351 total...</td>
    </tr>
    <tr>
      <th>test_name</th>
      <td>proportional_hazard_test</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>test_statistic</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">dependents_Yes</th>
      <th>km</th>
      <td>1.48</td>
      <td>0.22</td>
      <td>2.16</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.81</td>
      <td>0.37</td>
      <td>1.44</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">internetService_DSL</th>
      <th>km</th>
      <td>20.98</td>
      <td>&lt;0.005</td>
      <td>17.72</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>26.71</td>
      <td>&lt;0.005</td>
      <td>22.01</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">onlineBackup_Yes</th>
      <th>km</th>
      <td>17.80</td>
      <td>&lt;0.005</td>
      <td>15.31</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>17.47</td>
      <td>&lt;0.005</td>
      <td>15.07</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">techSupport_Yes</th>
      <th>km</th>
      <td>8.09</td>
      <td>&lt;0.005</td>
      <td>7.81</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>13.76</td>
      <td>&lt;0.005</td>
      <td>12.23</td>
    </tr>
  </tbody>
</table>


    
    
    1. Variable 'internetService_DSL' failed the non-proportional test: p-value is <5e-05.
    
       Advice: with so few unique values (only 2), you can include `strata=['internetService_DSL', ...]`
    in the call in `.fit`. See documentation in link [E] below.
    
       Bootstrapping lowess lines. May take a moment...
    
    
    2. Variable 'onlineBackup_Yes' failed the non-proportional test: p-value is <5e-05.
    
       Advice: with so few unique values (only 2), you can include `strata=['onlineBackup_Yes', ...]` in
    the call in `.fit`. See documentation in link [E] below.
    
       Bootstrapping lowess lines. May take a moment...
    
    
    3. Variable 'techSupport_Yes' failed the non-proportional test: p-value is 0.0002.
    
       Advice: with so few unique values (only 2), you can include `strata=['techSupport_Yes', ...]` in
    the call in `.fit`. See documentation in link [E] below.
    
    ---
    [A]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
    [B]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Bin-variable-and-stratify-on-it
    [C]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Introduce-time-varying-covariates
    [D]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Modify-the-functional-form
    [E]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
    





    [[<Axes: xlabel='rank-transformed time\n(p=0.3680)'>,
      <Axes: xlabel='km-transformed time\n(p=0.2232)'>],
     [<Axes: xlabel='rank-transformed time\n(p=0.0000)'>,
      <Axes: xlabel='km-transformed time\n(p=0.0000)'>],
     [<Axes: xlabel='rank-transformed time\n(p=0.0000)'>,
      <Axes: xlabel='km-transformed time\n(p=0.0000)'>],
     [<Axes: xlabel='rank-transformed time\n(p=0.0002)'>,
      <Axes: xlabel='km-transformed time\n(p=0.0044)'>]]




    
![png](/image/output_16_5.png)
    



    
![png](/image/output_16_6.png)
    



    
![png](/image/output_16_7.png)
    



    
![png](/image/output_16_8.png)
    


四个变量中只有一个p值 > 0.05，违反了四个变量中三个变量的比例风险假设

Schoenfeld 残差：用于验证模型是否符合比例风险假设。黑线越平坦，残差与时间越无关，也就越满足假设。能看到只有第一个变量的比较平坦

## 4.Accelerated Failure Time

### 目标
1. 将 Log-Logistic Accelerated Failure Time 模型拟合到 IBM 的 Telco 数据集
2. 解释 Accelerated Failure Time 模型的统计输出
3. 确定模型是否遵守基本假设

### Accelerated Failure Time
Accelerated Failure Time (AFT) 模型是一种参数生存分析模型，用于分析时间到事件的数据。与 Cox 比例风险模型不同，AFT 模型直接对生存时间进行建模，而不是对风险函数进行建模。\
AFT 模型的基本形式可以表示为：
$$T=exp(x^{T}\beta+\epsilon)$$
$T$是生存时间，$x$是协变量向量，$\beta$是模型参数向量\
$exp(x^{T}\beta)>1$时生存时间会缩短，反之延长

Kleinbaum和Klein在[《生存分析：自我学习文本》](https://link.springer.com/book/10.1007/978-1-4419-6646-9)中分享的加速失效时间模型的经典例子是狗的寿命。人们普遍认为，狗的衰老速度比人类快 7 倍。他们经历与我们相同的人生阶段，只是速度更快。使用下面的加速失效时间方程，如果我们将 A 组定义为人类，将 B 组定义为狗，那么加速因子 $\lambda$ 将为 7。同样，如果我们将 A 组定义为狗，将 B 组定义为人类，那么加速因子将为 1/7
$$ S_A(t) = S_B(t) * \lambda $$


```python
# 编码分类变量
encode_cols2 = ['partner', 'multipleLines', 'internetService', 'onlineSecurity', 'onlineBackup', 
               'deviceProtection', 'techSupport', 'paymentMethod']
encoded_pd2 = pd.get_dummies(telco_pd, columns=encode_cols2, prefix=encode_cols2, drop_first=False)
encoded_pd2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>seniorCitizen</th>
      <th>dependents</th>
      <th>tenure</th>
      <th>phoneService</th>
      <th>streamingTV</th>
      <th>streamingMovies</th>
      <th>contract</th>
      <th>paperlessBilling</th>
      <th>...</th>
      <th>onlineBackup_No</th>
      <th>onlineBackup_Yes</th>
      <th>deviceProtection_No</th>
      <th>deviceProtection_Yes</th>
      <th>techSupport_No</th>
      <th>techSupport_Yes</th>
      <th>paymentMethod_Bank transfer (automatic)</th>
      <th>paymentMethod_Credit card (automatic)</th>
      <th>paymentMethod_Electronic check</th>
      <th>paymentMethod_Mailed check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0.0</td>
      <td>No</td>
      <td>1.0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0.0</td>
      <td>No</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0.0</td>
      <td>No</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9305-CDSKC</td>
      <td>Female</td>
      <td>0.0</td>
      <td>No</td>
      <td>8.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1452-KIOVK</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>22.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
# 创建生存分析所需的数据集
survival_pd2 = encoded_pd2[['churn', 'tenure', 'partner_Yes', 'multipleLines_Yes', 
                          'internetService_DSL', 'onlineSecurity_Yes', 'onlineBackup_Yes', 
                          'deviceProtection_Yes', 'techSupport_Yes', 
                          'paymentMethod_Bank transfer (automatic)', 
                          'paymentMethod_Credit card (automatic)']]

# 将 'churn' 列转换为浮点类型
survival_pd2['churn'] = survival_pd2['churn'].astype('float')

import numpy as np
from lifelines import LogLogisticAFTFitter

# 初始化 LogLogisticAFTFitter，一个类
aft = LogLogisticAFTFitter()

# 拟合模型
aft.fit(survival_pd2, duration_col='tenure', event_col='churn')

# 打印模型摘要
print("Median Survival Time: {:.2f}".format(np.exp(aft.median_survival_time_)))
aft.print_summary()

# 绘制模型结果
aft.plot()
plt.show()
```

    /tmp/ipykernel_16780/1835610664.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      survival_pd2['churn'] = survival_pd2['churn'].astype('float')


    Median Survival Time: 135.51



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.LogLogisticAFTFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'tenure'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'churn'</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>3351</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>1556</td>
    </tr>
    <tr>
      <th>log-likelihood</th>
      <td>-6838.36</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2025-04-11 03:38:23 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">alpha_</th>
      <th>deviceProtection_Yes</th>
      <td>0.48</td>
      <td>1.62</td>
      <td>0.07</td>
      <td>0.35</td>
      <td>0.62</td>
      <td>1.41</td>
      <td>1.86</td>
      <td>0.00</td>
      <td>6.88</td>
      <td>&lt;0.005</td>
      <td>37.25</td>
    </tr>
    <tr>
      <th>internetService_DSL</th>
      <td>0.38</td>
      <td>1.47</td>
      <td>0.08</td>
      <td>0.23</td>
      <td>0.53</td>
      <td>1.26</td>
      <td>1.71</td>
      <td>0.00</td>
      <td>4.98</td>
      <td>&lt;0.005</td>
      <td>20.59</td>
    </tr>
    <tr>
      <th>multipleLines_Yes</th>
      <td>0.66</td>
      <td>1.94</td>
      <td>0.07</td>
      <td>0.53</td>
      <td>0.80</td>
      <td>1.70</td>
      <td>2.22</td>
      <td>0.00</td>
      <td>9.64</td>
      <td>&lt;0.005</td>
      <td>70.70</td>
    </tr>
    <tr>
      <th>onlineBackup_Yes</th>
      <td>0.81</td>
      <td>2.25</td>
      <td>0.07</td>
      <td>0.68</td>
      <td>0.95</td>
      <td>1.97</td>
      <td>2.59</td>
      <td>0.00</td>
      <td>11.63</td>
      <td>&lt;0.005</td>
      <td>101.50</td>
    </tr>
    <tr>
      <th>onlineSecurity_Yes</th>
      <td>0.86</td>
      <td>2.37</td>
      <td>0.09</td>
      <td>0.69</td>
      <td>1.03</td>
      <td>2.00</td>
      <td>2.80</td>
      <td>0.00</td>
      <td>10.12</td>
      <td>&lt;0.005</td>
      <td>77.60</td>
    </tr>
    <tr>
      <th>partner_Yes</th>
      <td>0.68</td>
      <td>1.97</td>
      <td>0.07</td>
      <td>0.55</td>
      <td>0.81</td>
      <td>1.73</td>
      <td>2.24</td>
      <td>0.00</td>
      <td>10.21</td>
      <td>&lt;0.005</td>
      <td>78.93</td>
    </tr>
    <tr>
      <th>paymentMethod_Bank transfer (automatic)</th>
      <td>0.74</td>
      <td>2.10</td>
      <td>0.09</td>
      <td>0.56</td>
      <td>0.92</td>
      <td>1.75</td>
      <td>2.51</td>
      <td>0.00</td>
      <td>8.05</td>
      <td>&lt;0.005</td>
      <td>50.07</td>
    </tr>
    <tr>
      <th>paymentMethod_Credit card (automatic)</th>
      <td>0.80</td>
      <td>2.22</td>
      <td>0.10</td>
      <td>0.61</td>
      <td>0.99</td>
      <td>1.84</td>
      <td>2.68</td>
      <td>0.00</td>
      <td>8.36</td>
      <td>&lt;0.005</td>
      <td>53.81</td>
    </tr>
    <tr>
      <th>techSupport_Yes</th>
      <td>0.69</td>
      <td>1.99</td>
      <td>0.09</td>
      <td>0.52</td>
      <td>0.86</td>
      <td>1.68</td>
      <td>2.36</td>
      <td>0.00</td>
      <td>7.90</td>
      <td>&lt;0.005</td>
      <td>48.37</td>
    </tr>
    <tr>
      <th>Intercept</th>
      <td>1.59</td>
      <td>4.91</td>
      <td>0.07</td>
      <td>1.46</td>
      <td>1.72</td>
      <td>4.32</td>
      <td>5.58</td>
      <td>0.00</td>
      <td>24.47</td>
      <td>&lt;0.005</td>
      <td>436.88</td>
    </tr>
    <tr>
      <th>beta_</th>
      <th>Intercept</th>
      <td>0.12</td>
      <td>1.13</td>
      <td>0.02</td>
      <td>0.08</td>
      <td>0.16</td>
      <td>1.08</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>5.71</td>
      <td>&lt;0.005</td>
      <td>26.42</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.73</td>
    </tr>
    <tr>
      <th>AIC</th>
      <td>13698.72</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>877.49 on 9 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>605.78</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_20_3.png)
    


每列的 p 值低于 < 0.005。因此，每列都具有统计显著性\
第1列为系数，以 internetService_DSL 为例，当客户将DSL作为其互联网服务时，客户的留存时间会比不使用DSL的加速 1.47 倍，留存时间更短


```python
from lifelines import KaplanMeierFitter

# 初始化 KaplanMeierFitter
kmf = KaplanMeierFitter()

# 提取生存时间和生存状态
T = telco_pd['tenure']
C = telco_pd['churn'].astype(float)

# 拟合模型
kmf.fit(T, C)

# 绘制生存曲线
kmf.plot(title='Kaplan-Meier Survival Curve: Population level')
plt.show()
```


    
![png](output_22_0.png)
    


### 验证模型是否符合假设


```python
def plot_km_logOdds(col):
    ax = plt.subplot(111)
    for r in telco_pd[col].unique():
        ix = telco_pd[col] == r
        kmf.fit(T[ix], C[ix], label=r)
        sf = kmf.survival_function_
        sf['failureOdds'] = (np.log(1 - sf)) / sf
        sf['logTime'] = np.log(sf.index)
        plt.plot(sf['logTime'], sf['failureOdds'], label=r)
    plt.title(f'Kaplan-Meier Log-Odds Curves by {col}')
    plt.legend()
    plt.show()

# 绘制不同特征的生存曲线
plot_km_logOdds('partner')
plot_km_logOdds('multipleLines')
plot_km_logOdds('internetService')
plot_km_logOdds('onlineSecurity')
plot_km_logOdds('onlineBackup')
plot_km_logOdds('deviceProtection')
plot_km_logOdds('techSupport')
plot_km_logOdds('paymentMethod')
```


    
![png](output_24_0.png)
    



    
![png](output_24_1.png)
    



    
![png](output_24_2.png)
    



    
![png](output_24_3.png)
    



    
![png](output_24_4.png)
    



    
![png](output_24_5.png)
    



    
![png](output_24_6.png)
    



    
![png](output_24_7.png)
    


x轴为 $log(t)$，y轴为 $log \frac{1-S(t)}{S(t)} $

直线的斜率可以提供关于分布形状的信息。斜率的绝对值越大，表示随着世界增加，生存概率迅速下降\
大部分图中的线条都相对笔直。有一些偏差，但总体上还不错。这意味着选择 log-logistic 是一个良好的模型\
大部分图中的线条并不平行。这意味着模型不太符合Accelerated Failure Time的假设

## 5.Customer Lifetime Value

### 目标
1. 将 Cox Proportional Hazard 模型拟合到 IBM 的 Telco 数据集。
2. 解释 Cox 比例风险模型的统计输出。
3. 确定模型是遵循还是违反比例风险假设。


```python
import pandas as pd

# 编码分类变量
encode_cols3 = ['dependents', 'internetService', 'onlineBackup', 'techSupport', 'paperlessBilling']
encoded_pd3 = pd.get_dummies(telco_pd, columns=encode_cols3, prefix=encode_cols3, drop_first=False)

# 创建生存分析所需的数据集
survival_pd3 = encoded_pd3[['churn', 'tenure', 'dependents_Yes', 'internetService_DSL', 'onlineBackup_Yes',
                            'techSupport_Yes']]

# 将 'churn' 列转换为浮点类型
survival_pd3['churn'] = survival_pd3['churn'].astype('float')
```

    /tmp/ipykernel_16780/2671776062.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      survival_pd3['churn'] = survival_pd3['churn'].astype('float')


### 拟合比例风险模型


```python
from lifelines.fitters.coxph_fitter import CoxPHFitter

# 初始化 Cox 比例风险模型
cph = CoxPHFitter(alpha=0.05)
cph.fit(survival_pd3, duration_col='tenure', event_col='churn')

# 打印模型摘要
cph.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'tenure'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'churn'</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>3351</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>1556</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-11315.95</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2025-04-11 05:11:16 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dependents_Yes</th>
      <td>-0.33</td>
      <td>0.72</td>
      <td>0.07</td>
      <td>-0.47</td>
      <td>-0.19</td>
      <td>0.63</td>
      <td>0.83</td>
      <td>0.00</td>
      <td>-4.64</td>
      <td>&lt;0.005</td>
      <td>18.12</td>
    </tr>
    <tr>
      <th>internetService_DSL</th>
      <td>-0.22</td>
      <td>0.80</td>
      <td>0.06</td>
      <td>-0.33</td>
      <td>-0.10</td>
      <td>0.72</td>
      <td>0.90</td>
      <td>0.00</td>
      <td>-3.68</td>
      <td>&lt;0.005</td>
      <td>12.07</td>
    </tr>
    <tr>
      <th>onlineBackup_Yes</th>
      <td>-0.78</td>
      <td>0.46</td>
      <td>0.06</td>
      <td>-0.89</td>
      <td>-0.66</td>
      <td>0.41</td>
      <td>0.52</td>
      <td>0.00</td>
      <td>-13.13</td>
      <td>&lt;0.005</td>
      <td>128.37</td>
    </tr>
    <tr>
      <th>techSupport_Yes</th>
      <td>-0.64</td>
      <td>0.53</td>
      <td>0.08</td>
      <td>-0.79</td>
      <td>-0.49</td>
      <td>0.46</td>
      <td>0.61</td>
      <td>0.00</td>
      <td>-8.48</td>
      <td>&lt;0.005</td>
      <td>55.36</td>
    </tr>
  </tbody>
</table><br><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.64</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>22639.90</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>337.77 on 4 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>236.24</td>
    </tr>
  </tbody>
</table>
</div>


### 创建交互式小部件widget


```python
# 使用ipywidgets 库来创建交互式小部件
import ipywidgets as widgets
from IPython.display import display

cols = ['dependents_Yes', 'internetService_DSL', 'onlineBackup_Yes', 'techSupport_Yes', 'partner_Yes', 
        'internal rate of return']

widgets_dict = {}# 用于存储创建的小部件
for col in cols:
    # 如果列名是 'internal rate of return'，创建一个文本框（widgets.Text），默认值为 '0.10'
    if col == 'internal rate of return':
        widgets_dict[col] = widgets.Text(value='0.10', description=col) 
    # 对于其他列，创建一个下拉菜单（widgets.Dropdown），选项为 ['0', '1']，默认值为 '0'
    else:
        widgets_dict[col] = widgets.Dropdown(options=['0', '1'], value='0', description=col)

# 显示小部件
for widget in widgets_dict.values():
    display(widget)
```


    Dropdown(description='dependents_Yes', options=('0', '1'), value='0')



    Dropdown(description='internetService_DSL', options=('0', '1'), value='0')



    Dropdown(description='onlineBackup_Yes', options=('0', '1'), value='0')



    Dropdown(description='techSupport_Yes', options=('0', '1'), value='0')



    Dropdown(description='partner_Yes', options=('0', '1'), value='0')



    Text(value='0.10', description='internal rate of return')



```python
# 根据通过小部件输入的值，计算并绘制生存概率曲线和累计净现值（NPV）的柱状图
# 从通过小部件输入的值中提取数据，并将其转换为 Pandas DataFrame
def get_widget_values():
    # 表示将字典的键作为行索引，T 表示转置 DataFrame，使其每一行代表一个样本
    widget_dict = {col: widget.value for col, widget in widgets_dict.items()}
    return pd.DataFrame.from_dict(widget_dict, orient='index').T

# 根据输入的值，计算生存概率、平均预期月利润、平均预期月利润的净现值（NPV）和累计净现值
def get_payback_df():
    df = get_widget_values()
    irr = float(df['internal rate of return'][0]) / 12 # 年利率转为月利率
    cohort_df = pd.concat([pd.DataFrame([1.00]), round(cph.predict_survival_function(df), 2)])\
    .rename(columns={0: 'Survival Probability'})
    cohort_df['Contract Month'] = cohort_df.index.astype('int')
    cohort_df['Monthly Profit for the Selected Plan'] = 30 # 假设每月的利润为 30
    # 生存概率乘以每月利润
    cohort_df['Avg Expected Monthly Profit'] = round(cohort_df['Survival Probability'] * 
                                                     cohort_df['Monthly Profit for the Selected Plan'], 2)
    # 平均预期月利润的净现值
    cohort_df['NPV of Avg Expected Monthly Profit'] = round(cohort_df['Avg Expected Monthly Profit'] / 
                                                            ((1 + irr) ** cohort_df['Contract Month']), 2)
    # 累计净现值
    cohort_df['Cumulative NPV'] = cohort_df['NPV of Avg Expected Monthly Profit'].cumsum()
    # 合同期限，从 1 开始计数
    cohort_df['Contract Month'] = cohort_df['Contract Month'] + 1
    return cohort_df[['Contract Month', 'Survival Probability', 'Monthly Profit for the Selected Plan', 
                      'Avg Expected Monthly Profit', 'NPV of Avg Expected Monthly Profit', 
                      'Cumulative NPV']].set_index('Contract Month')

# 绘制柱状图
import seaborn as sns
import matplotlib.pyplot as plt
# 获取数据
payback_df = get_payback_df()
months = ['12 Months', '24 Months', '36 Months']
cumulative_npv = payback_df.iloc[[11, 23, 35], :]['Cumulative NPV']

# 创建一个 Pandas DataFrame
plot_data = pd.DataFrame({
    'Contract Month': months,
    'Cumulative NPV': cumulative_npv
})

# 绘制柱状图
ax = sns.barplot(x='Contract Month', y='Cumulative NPV', data=plot_data)
plt.show()
```


    
![png](output_31_0.png)
    


以第一个柱子为例，柱子的高度表示客户留存12个月时所有预期利润的净现值之和


```python
# 绘制生存概率曲线
sns.lineplot(x=payback_df.index, y=payback_df['Survival Probability'])
plt.show()
```


    
![png](output_33_0.png)
    


## 6.Reference
内容主要来自[Survival Analysis for Churn and Lifetime Value](https://www.databricks.com/solutions/accelerators/survival-analysis-for-churn-and-lifetime-value)

对各种方法的介绍和部分与原文不同的代码由LLM生成
