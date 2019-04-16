# 用户自定义表生成函数

一般的用户自定义函数，比如`concat()`，接受单个输入行并输出单个输出行。与其相反的是表生成函数可以将单个输入行转换为多个输出行。


## explode(array)
其作用是将array展开为多行，每一行是单个元素。  
用法示例：  
```sql
select explode(array('A','B','C'));
select explode(array('A','B','C')) as col;
select tf.* from (select 0) t lateral view explode(array('A','B','C')) tf;
select tf.* from (select 0) t lateral view explode(array('A','B','C')) tf as col;

```

结果：  
```
col
A
B
C
```


## explode(map)  
其作用是将map展开为多行，每一行是一个key-value键值对。

用法示例：  
```sql
select explode(map('A',10,'B',20,'C',30));
select explode(map('A',10,'B',20,'C',30)) as (key,value);
select tf.* from (select 0) t lateral view explode(map('A',10,'B',20,'C',30)) tf;
select tf.* from (select 0) t lateral view explode(map('A',10,'B',20,'C',30)) tf as key,value;
```

结果：  
```
key value
A	10
B	20
C	30
```

## posexplode(array)

其作用是在展开array的同时附加一个整数类型的位置列，每一行由 位置和其对应的值构成。

用法示例：  
```sql
select posexplode(array('A','B','C'));
select posexplode(array('A','B','C')) as (pos,val);
select tf.* from (select 0) t lateral view posexplode(array('A','B','C')) tf;
select tf.* from (select 0) t lateral view posexplode(array('A','B','C')) tf as pos,val;

```

结果：  
```
pos val
0	A
1	B
2	C
```

## inline(array<struct>)
其作用是将结构体数组展开为多行，每一行对应一个结构体。

用法示例：  
```sql
select inline(array(struct('A',10,date '2015-01-01'),struct('B',20,date '2016-02-02')));
select inline(array(struct('A',10,date '2015-01-01'),struct('B',20,date '2016-02-02'))) as (col1,col2,col3);
select tf.* from (select 0) t lateral view inline(array(struct('A',10,date '2015-01-01'),struct('B',20,date '2016-02-02'))) tf;
select tf.* from (select 0) t lateral view inline(array(struct('A',10,date '2015-01-01'),struct('B',20,date '2016-02-02'))) tf as col1,col2,col3;
```

结果：
```
col1    col2    col3
A	10	2015-01-01
B	20	2016-02-02
```

## stack(int r,v_1,v_2,...,v_k)

其作用是将k个值分解为r行，每一行有 k/r 列，r必须是常数。

用法示例：
```sql
select stack(2,'A',10,date '2015-01-01','B',20,date '2016-01-01');
select stack(2,'A',10,date '2015-01-01','B',20,date '2016-01-01') as (col0,col1,col2);
select tf.* from (select 0) t lateral view stack(2,'A',10,date '2015-01-01','B',20,date '2016-01-01') tf;
select tf.* from (select 0) t lateral view stack(2,'A',10,date '2015-01-01','B',20,date '2016-01-01') tf as col0,col1,col2;
```

结果：  
```sql
col0 col1 col2
A	10	2015-01-01
B	20	2016-01-01
```

## json_tuple(string jsonStr,string k_1,...,string k_n)

其作用是接收一个json字符串以及一组key的名字作为输入，从json中解析key对应的value并返回。

用法示例：
```sql
select json_tuple('{"id":1,"name":"peter","gender":"male"}','id','name') as (id,name);
```

## parse_url_typle(string urlStr,string p_1,...,string p_n)

其作用是与`parse_url`函数类似，都是解析url，不同之处在于它可以同时抽取给定url中的多个元素。它接受一个url以及n个url片段名作为输入，并返回一组由n个值组成的tuple。有效的url片段名有：HOST, PATH, QUERY, REF, PROTOCOL, AUTHORITY, FILE, USERINFO, QUERY:<KEY>

用法示例：  
```sql
SELECT parse_url_tuple('http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1', 'HOST', 'PATH', 'QUERY', 'QUERY:k1', 'QUERY:k2');
```

结果：  
```
facebook.com	/path1/p.php	k1=v1&k2=v2	v1	v2
```
