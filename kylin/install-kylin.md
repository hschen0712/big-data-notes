# Kylin安装

## 环境说明

目前公司的大数据集群安装的Hadoop版本是Cloudera的cdh5.6，具体对应到各个组件上的版本是：
- Hbase 1.0.0-cdh5.6.0
- Hive 1.2.1-cdh5.6.0
- Spark 2.2.0
- Hadoop 2.6.0-cdh5.6.0

## 环境变量

Kylin官网没有适用于CDH 5.6的安装包，但通过查阅资料，发现CDH5.6可以兼容 `apache-kylin-2.6.1-bin-cdh57.tar.gz` 安装包。开始安装前，请确保环境变量已配置好，贴一份我的环境变量

```shell
export HADOOP_HOME=/opt/cdh5.6/hadoop
export HIVE_HOME=/opt/cdh5.6/hive
export HADOOP_HOME=/opt/cdh5.6/hadoop
export SQOOP_HOME=/opt/cdh5.6/sqoop
export HCAT_HOME=/opt/cdh5.6/hive/hcatalog
export SCALA_HOME=/opt/scala
export SPARK_HOME=/opt/spark
export JAVA_HOME=/opt/jdk
export HBASE_HOME=/opt/cdh5.6/hbase
```

## 安装过程

1.下载安装包

```shell
wget http://apache.01link.hk/kylin/apache-kylin-2.6.1/apache-kylin-2.6.1-bin-cdh57.tar.gz
```

2.解压

```shell
tar zxf apache-kylin-2.6.1-bin-cdh57.tar.gz
```

3.将kylin目录加入到环境变量
打开~/.bashrc，加入 `export KYLIN_HOME=/home/work/apache-kylin-2.6.1-bin-cdh57`，接着执行 `source ~/.bashrc` 命令使环境变量生效

4.启动kylin

```shell
cd $KYLIN_HOME/bin
sh ./kylin.sh start
```

到此，理论上kylin就已经正常启动了（如下图）。
![](http://pq079rf22.bkt.clouddn.com/kylin-install.png)

但是，实际部署的时候会遇到各种各样的坑，下面列举出笔者在安装过程中遇到的坑。

## 安装过程遇到的各种疑难杂症

1.无法在hdfs创建/kylin目录
原因分析：这个问题发生的原因通常是我们创建目录使用的用户不是超级用户，故没有权限。
解决办法：  
打开 `$KYLIN_HOME/conf/kylin.properties`，将 `kylin.env.hdfs-working-dir` 选项修改为 `/tmp/kylin` 或者其他有读写权限的目录即可。

2.Web UI无法打开
原因分析：kylin web界面默认的服务器端口是7070，而服务器的防火墙只允许8000以后的端口访问
解决办法：
打开 `$KYLIN_HOME/tomcat/conf/server.xml`，将默认端口改为8077（只要8000以后即可）:
```shell
    <Connector port="8077" protocol="HTTP/1.1"
    connectionTimeout="20000"
    redirectPort="7443"
    compression="off"
    compressionMinSize="2048"
    noCompressionUserAgents="gozilla,traviata"
    compressableMimeType="text/html,text/xml,text/javascript,application/javascript,application/json,text/css,text/plain"
    />
```

接着我们可以用nc命令测试是否能从外网访问端口，如果出现以下提示就说明可以正常访问：
```shell
nc -vv xxx.xxx.xxx.xxx 8077
found 0 associations
found 1 connections:
1:	flags=82<CONNECTED,PREFERRED>
outif utun2
src yyy.yyy.yyy.yyy port 63252
dst xxx.xxx.xxx.xxxx port 8077
rank info not available
TCP aux info available

Connection to xxx.xxx.xxx.xxx port 8077 [tcp/*] succeeded!
``` 

我们再来看一下web界面能否正常访问

3.spark assembly lib not found
原因分析：这是由于使用的kylin版本与大数据集群的spark版本不兼容，笔者最初安装的kylin版本只支持spark1.6版本
解决办法：下载一个合适的kylin版本并重新安装
