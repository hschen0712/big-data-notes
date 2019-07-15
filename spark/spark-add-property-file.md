# Spark submit如何指定自己的配置文件

## 问题背景

通过spark-submit命令提交任务，在提交时我们想给spark程序传入额外的参数，比如输入、输出路径，应该怎么做？

## 解决方案

我们可以编写一个与java中类似的`property`文件，格式如下：

```
spark.myapp.input /input/path
spark.myapp.output /output/path
```

然后在提交任务时通过`--properties-file`参数来指定property文件

```
$SPARK_HOME/bin/spark-submit --properties-file  mypropsfile.conf
```

在spark程序中，可以通过以下方式获取参数

```
spark.conf.get("spark.myapp.input")
spark.conf.get("spark.myapp.output")
```

此外，我们也可以通过在程序中使用

```
spark.sparkContext.addFile(dir, true)
```

来添加配置文件，其中`dir`是配置文件所在的目录，第二个参数将`recursive`属性设置为`true`，将允许递归读取该目录下的配置文件。


## 参考

https://stackoverflow.com/questions/31115881/how-to-load-java-properties-file-and-use-in-spark/42816733  
https://stackoverflow.com/questions/38879478/sparkcontext-addfile-vs-spark-submit-files
