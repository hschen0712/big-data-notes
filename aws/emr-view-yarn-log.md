# AWS EMR查看yarn日志

## 问题背景

通过EC2执行`aws emr create-cluster`命令来创建EMR集群，并指定了`--auto-terminate`参数使得任务执行完成后集群自动终止。如果步骤中有spark任务，且中途报错了，但由于集群已经被终止，无法通过`yarn logs -applicationId <application_id>`命令来查看错误日志，这种情况下该去哪里查看yarn日志呢？

## 解决方案

EMR的yarn日志都存储在S3的应用程序容器日志里面，其路径是`s3://aws-logs-<aws_id>-<aws_region>/elasticmapreduce/<cluster_id>/containers/<application_id>/`

## 参考

https://docs.aws.amazon.com/zh_cn/emr/latest/ManagementGuide/emr-manage-view-web-log-files.html
