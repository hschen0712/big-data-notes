# MLlib梯度提升树源码解读


## GBT的实现

梯度提升树的实现位于 `org.apache.spark.ml.tree.impl`包下的`GradientBoostedTrees.scala` 文件中。  
下面对实现过程中用到的方法进行介绍。

### computeInitialPredictionAndError

计算梯度提升树模型第一棵树的初始预测值以及误差

```scala

  /**
   * Compute the initial predictions and errors for a dataset for the first
   * iteration of gradient boosting.
   * @param data: training data.
   * @param initTreeWeight: learning rate assigned to the first tree.
   * @param initTree: first DecisionTreeModel.
   * @param loss: evaluation metric.
   * @return an RDD with each element being a zip of the prediction and error
   *         corresponding to every sample.
   */
  def computeInitialPredictionAndError(
      data: RDD[LabeledPoint], //训练数据
      initTreeWeight: Double, //第一棵树的权重(学习率)
      initTree: DecisionTreeRegressionModel, //第一棵决策树模型
      loss: OldLoss //损失函数
  ): RDD[(Double, Double)] = {
    data.map { lp =>
      val pred = updatePrediction(lp.features, 0.0, initTree, initTreeWeight) //更新预测值
      val error = loss.computeError(pred, lp.label) //计算损失函数值
      (pred, error)
    }
  }

```


### updatePredictionError  

基于上一轮迭代的预测值，以及新一轮迭代训练的基学习器及其权重，更新预测值及损失

```scala

  /**
   * Update a zipped predictionError RDD
   * (as obtained with computeInitialPredictionAndError)
   * @param data: training data.
   * @param predictionAndError: predictionError RDD
   * @param treeWeight: Learning rate.
   * @param tree: Tree using which the prediction and error should be updated.
   * @param loss: evaluation metric.
   * @return an RDD with each element being a zip of the prediction and error
   *         corresponding to each sample.
   */
  def updatePredictionError(
      data: RDD[LabeledPoint], //训练数据
      predictionAndError: RDD[(Double, Double)], //上一轮的预测值及损失
      treeWeight: Double, //树的权重，即学习率
      tree: DecisionTreeRegressionModel, //新加入的模型
      loss: OldLoss //损失函数
  ): RDD[(Double, Double)] = {

    // zip用于将两个RDD组成kev-value形式的RDD，key来自第一个RDD，value来自第二个RDD，需要注意的是两个RDD的分区数以及元素个数都必须相同。其类型是RDD[(LabeledPoint,(Double,Double))]，其元素是(lp,(pred,error))
    val newPredError = data.zip(predictionAndError).mapPartitions { iter =>
      iter.map { case (lp, (pred, error)) =>
        val newPred = updatePrediction(lp.features, pred, tree, treeWeight)
        val newError = loss.computeError(newPred, lp.label)
        (newPred, newError)
      }
    }
    newPredError
  }
```


###  updatePrediction

基于上一轮迭代的预测值，以及新一轮迭代训练的基学习器及其权重，更新预测值

```scala

  /**
   * Add prediction from a new boosting iteration to an existing prediction.
   *
   * @param features Vector of features representing a single data point.
   * @param prediction The existing prediction.
   * @param tree New Decision Tree model.
   * @param weight Tree weight.
   * @return Updated prediction.
   */
  def updatePrediction(
      features: Vector,
      prediction: Double,
      tree: DecisionTreeRegressionModel,
      weight: Double): Double = {
    //predictImpl从树的根节点递归地进行遍历到叶子节点，计算预测值
    prediction + tree.rootNode.predictImpl(features).prediction * weight
  }

```


### boost

训练GBDT模型的核心代码位于boost方法中。  

```scala

  /**
   * Internal method for performing regression using trees as base learners.
   * @param input training dataset
   * @param validationInput validation dataset, ignored if validate is set to false.
   * @param boostingStrategy boosting parameters
   * @param validate whether or not to use the validation dataset.
   * @param seed Random seed.
   * @return tuple of ensemble models and weights:
   *         (array of decision tree models, array of model weights)
   */
  def boost(
      input: RDD[LabeledPoint], //训练集
      validationInput: RDD[LabeledPoint], //验证集
      boostingStrategy: OldBoostingStrategy, //boosting算法的参数
      validate: Boolean, //是否使用验证集
      seed: Long, //随机数种子
      featureSubsetStrategy: String): //特征子集策略
(Array[DecisionTreeRegressionModel], Array[Double]) = {
    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")

    boostingStrategy.assertValid()

    // Initialize gradient boosting parameters
	//numIterations实际上是GBDT模型中baseLearner的数量
    val numIterations = boostingStrategy.numIterations 
    val baseLearners = new Array[DecisionTreeRegressionModel](numIterations)
   	// baseLearner的权重
    val baseLearnerWeights = new Array[Double](numIterations)
  	// 损失函数
    val loss = boostingStrategy.loss
  	// 学习率
    val learningRate = boostingStrategy.learningRate

  	// 设置单棵树的参数
    val treeStrategy = boostingStrategy.treeStrategy.copy
  	// early stopping的阈值
    val validationTol = boostingStrategy.validationTol
  	// 回归树
    treeStrategy.algo = OldAlgo.Regression
  	// impurity为方差
    treeStrategy.impurity = OldVariance
    treeStrategy.assertValid()

    // Cache input
    val persistedInput = if (input.getStorageLevel == StorageLevel.NONE) {
      input.persist(StorageLevel.MEMORY_AND_DISK)
      true
    } else {
      false
    }

    // Prepare periodic checkpointers
    val predErrorCheckpointer = new PeriodicRDDCheckpointer[(Double, Double)](
      treeStrategy.getCheckpointInterval, input.sparkContext)
    val validatePredErrorCheckpointer = new PeriodicRDDCheckpointer[(Double, Double)](
      treeStrategy.getCheckpointInterval, input.sparkContext)

    timer.stop("init")

    logDebug("##########")
    logDebug("Building tree 0")
    logDebug("##########")

    // 训练第一棵树
    timer.start("building tree 0")
    val firstTree = new DecisionTreeRegressor().setSeed(seed)
   	// 训练回归树模型，底层调用了RandomForest的train方法
    val firstTreeModel = firstTree.train(input, treeStrategy, featureSubsetStrategy)
    val firstTreeWeight = 1.0
  	// 将决策树模型加入baseLearner的数组
    baseLearners(0) = firstTreeModel
    baseLearnerWeights(0) = firstTreeWeight

  	// 计算第一棵树在训练集上的预测值及损失函数值
    var predError: RDD[(Double, Double)] =
      computeInitialPredictionAndError(input, firstTreeWeight, firstTreeModel, loss)
    predErrorCheckpointer.update(predError)
    logDebug("error of gbt = " + predError.values.mean())

    // Note: A model of type regression is used since we require raw prediction
    timer.stop("building tree 0")

  	// 计算第一棵树在验证集上的预测值及误差
    var validatePredError: RDD[(Double, Double)] =
      computeInitialPredictionAndError(validationInput, firstTreeWeight, firstTreeModel, loss)
    if (validate) validatePredErrorCheckpointer.update(validatePredError)
  	// 最优验证集误差
    var bestValidateError = if (validate) validatePredError.values.mean() else 0.0
  	// 最优验证集误差对应的迭代轮数
    var bestM = 1 

    var m = 1
    var doneLearning = false
    while (m < numIterations && !doneLearning) {
      // 生成下一轮迭代的训练数据，特征不变，目标为损失函数的负梯度(伪残差)
      val data = predError.zip(input).map { case ((pred, _), point) =>
        LabeledPoint(-loss.gradient(pred, point.label), point.features)
      }

      timer.start(s"building tree $m")
      logDebug("###################################################")
      logDebug("Gradient boosting tree iteration " + m)
      logDebug("###################################################")

      val dt = new DecisionTreeRegressor().setSeed(seed + m)
      val model = dt.train(data, treeStrategy, featureSubsetStrategy)
      timer.stop(s"building tree $m")
      // Update partial model
      baseLearners(m) = model
      // 以学习率作为第m个baseLearner的权重，这对于除了平方误差以外的损失函数是错误的，权重应该是针对特定的损失函数优化得出，这样做虽然不是最优，但也是相对合理的。
      baseLearnerWeights(m) = learningRate
	  // 计算baseLearner的预测值及损失
      predError = updatePredictionError(
        input, predError, baseLearnerWeights(m), baseLearners(m), loss)
      predErrorCheckpointer.update(predError)
      logDebug("error of gbt = " + predError.values.mean())

      if (validate) {
        // Stop training early if
        // 1. Reduction in error is less than the validationTol or
        // 2. If the error increases, that is if the model is overfit.
        // We want the model returned corresponding to the best validation error.
		// 验证集上的预测及损失
        validatePredError = updatePredictionError(
          validationInput, validatePredError, baseLearnerWeights(m), baseLearners(m), loss)
        validatePredErrorCheckpointer.update(validatePredError)
        //当前验证集上的平均误差
        val currentValidateError = validatePredError.values.mean()
        //如果最优验证集误差与当前验证集误差的差值小于阈值，则提前停止训练
        if (bestValidateError - currentValidateError < validationTol * Math.max(
          currentValidateError, 0.01)) {
          doneLearning = true
        } else if (currentValidateError < bestValidateError) {
          //如果当前验证集误差优于最优验证集误差，则进行更新
          bestValidateError = currentValidateError
          bestM = m + 1
        }
      }
      m += 1
    }

    timer.stop("total")

    logInfo("Internal timing for DecisionTree:")
    logInfo(s"$timer")

    predErrorCheckpointer.unpersistDataSet()
    predErrorCheckpointer.deleteAllCheckpoints()
    validatePredErrorCheckpointer.unpersistDataSet()
    validatePredErrorCheckpointer.deleteAllCheckpoints()
    if (persistedInput) input.unpersist()

    if (validate) {
      (baseLearners.slice(0, bestM), baseLearnerWeights.slice(0, bestM))
    } else {
      (baseLearners, baseLearnerWeights)
    }
  }
```
通过阅读代码，可以知道梯度提升树模型的构建过程是串行的。每一轮迭代中，利用当前模型来更新训练集的残差，并作为下一轮训练的输入。当迭代完成或验证集误差不再减少时，即完成整个模型的训练过程。


### run

run方法对GBT的训练逻辑进行了封装。可以通过algo指定任务的类型：分类或回归。如果是分类，需要将其转化为回归问题，具体方法是将label从{0,1}映射到{-1,1}进行训练，预测时根据预测值的符号决定输出的类别，如果预测值大于0，则输出类别为+1；反之输出类别为-1。

```scala
  def run(
      input: RDD[LabeledPoint],
      boostingStrategy: OldBoostingStrategy,
      seed: Long,
      featureSubsetStrategy: String): (Array[DecisionTreeRegressionModel], Array[Double]) = {
    val algo = boostingStrategy.treeStrategy.algo
    algo match {
      case OldAlgo.Regression =>
        GradientBoostedTrees.boost(input, input, boostingStrategy, validate = false,
          seed, featureSubsetStrategy)
      case OldAlgo.Classification =>
        // 将label从{0,1}映射到{-1,1}
        val remappedInput = input.map(x => new LabeledPoint((x.label * 2) - 1, x.features))
        GradientBoostedTrees.boost(remappedInput, remappedInput, boostingStrategy, validate = false,
          seed, featureSubsetStrategy)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by gradient boosting.")
    }
  }
```

如果需要在训练时通过额外的验证集来验证模型，可以通过`runWithValidation`方法


## 思考

- 为什么要用学习率作为每个baseLearner的权重？  
答：这是因为Spark在实现梯度提升树时，使用了Shrinkage技术，它是一种正则化的方法，为了防止模型过拟合。具体的做法是在更新模型时，对新加入的基学习器乘以一个学习率(通常是个比较小的数，比如0.1)。经验分析发现，引入Shrinkage技术后，模型的泛化能力得到了显著的改善，但这也意味着训练时间的增加，因为我们需要更多的迭代次数去拟合目标。

