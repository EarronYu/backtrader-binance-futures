# A Binance Futures trading and backtesting framework based on rodrigo-brito's backtrader-binance-bot

### Special thanks to rodrigo-brito

### Installation

Activating [Virtualenv](https://virtualenv.pypa.io/en/latest/)
```
make init
source venv/bin/activate
```

Installing dependencies
```
make install
```

Start application
```
./main.py
```

## Results

![alt text](screenshot.png "Backtrader Simulation")


```
Starting Portfolio Value: 100000.00
Final Portfolio Value: 119192.61

Profit 19.193%
Trade Analysis Results:
               Total Open     Total Closed   Total Won      Total Lost     
               0              10             7              3              
               Strike Rate    Win Streak     Losing Streak  PnL Net        
               1              5              2              19192.61       
SQN: 1.75
```

## To do list

多策略，多数据源，多时间周期的backtrader框架

自身带有前进式回测分析WFA

策略能够方便的加入一些仓位控制方法（简单如凯利公式）
#

## 至今的所有修改进度

20211129 
多策略部分除了分配持仓比例部分未写好其它部分已经完全搞定（还未debug）

目前能加入多币种多时间周期（未debug）

walk forward analyze 完全照搬 Ugur Akyol‘s WalkForwardWorkSheet，目前正在阅读代码还未完全的套用进框架。

CURTIS MILLER'S walk forward analyze 我也正在看，但不知道跟Akyol两者之间谁更好

当WFA的回测写好之后甚至不需要将其加入实盘框架，本地手动回测就好，毕竟服务器的计算能力是真的无法承担多策略不同参数的寻优

接触到了新的优化模块blackbox
