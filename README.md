# Car Evaluation

![Car Test](https://s.hswstatic.com/gif/automakers-test-long-term-durability-1.jpg)

Predicting model implemented in Apache Spark MLlib to evaluate cars depending on diverse attributes, such as safety, number of doors or maintenance cost.

## Data

We’ll use a well-known dataset from the UCI Machine Learning Repository, called Car Evaluation Dataset. This dataset is comprised of 1728 instances and 6 attributes. Fortunately, it doesn’t have any missing values.

This is a multiclass problem. Give a set of categorical input features, we’ll predict a label from a pool of four classes.

Let’s see the features in depth:


+ buying: Buying price of the car. Possible values:
    - vhigh: Very high.
    - high: High.
    - med: Medium.
    - low: Low.
+ maint: Maintenance price. Possible values:
     - vhigh: Very high.
     - high: High.
     - med: Medium.
     - low: Low.
+ doors: Number of doors. Possible values:
     -  2: Car has 2 doors.
     -  3: Car has 3 doors.
     -  4: Car has 4 doors.
     -   5more: Car has 5 doors or more.
+ persons: Car’s capacity. Possible values:
     -   2: Car can carry 2 people.
     -   4: Car can carry 4 people.
     -   more: Car can carry more than 4 people.
+ lug_boot: The size of the luggage boot. Possible values:
     -   small: Small size.
     -   med: Medium size.
     -   big: Big size.
+ safety: Estimated car’s safety. Possible values:
     -   low: Not very safe.
     -   med: Relatively safe.
     -   high: Very safe.

The target value is also categorical and measures the overall acceptability of the car. The possible values are:

 + unacc: Unacceptable.
 + acc: Acceptable.
+  good: Good.
  +  vgood: Very good.


## Run

You'll need `sbt 1.1.1.1` and `Scala 2.11.11`.

`sbt run`

## More info

Check my related blog post [here](http://datasmarts.net/2018/03/17/building-a-car-acceptance-classifier-in-spark-mllib/). ;)