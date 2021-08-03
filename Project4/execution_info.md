- Using the provided parameters, the program produced the output:
```
Loaded 60000 examples
Loaded 10000 examples
TRAINING FINISHED!

TRAINING SAMPLES CONFUSION MATRIX:
```

|||||||||||
|------	|------	|------	|------	|------	|------	|------	|------	|------	|------	|
| 5868 	| 25   	| 71   	| 55   	| 21   	| 0    	| 104  	| 0    	| 35   	| 0    	|
| 4    	| 5872 	| 3    	| 11   	| 2    	| 1    	| 6    	| 0    	| 4    	| 0    	|
| 25   	| 3    	| 5711 	| 8    	| 98   	| 1    	| 95   	| 0    	| 19   	| 0    	|
| 36   	| 77   	| 37   	| 5837 	| 47   	| 0    	| 49   	| 0    	| 15   	| 0    	|
| 19   	| 9    	| 129  	| 55   	| 5778 	| 0    	| 63   	| 0    	| 14   	| 0    	|
| 3    	| 1    	| 2    	| 4    	| 3    	| 5957 	| 1    	| 16   	| 7    	| 12   	|
| 24   	| 6    	| 27   	| 22   	| 39   	| 0    	| 5659 	| 0    	| 13   	| 0    	|
| 1    	| 1    	| 0    	| 1    	| 0    	| 26   	| 0    	| 5926 	| 12   	| 48   	|
| 19   	| 5    	| 20   	| 5    	| 12   	| 6    	| 21   	| 12   	| 5877 	| 0    	|
| 1    	| 1    	| 0    	| 2    	| 0    	| 9    	| 2    	| 46   	| 4    	| 5940 	|

```
TESTING SAMPLES CONFUSION MATRIX:
```

|||||||||||
|------	|------	|------	|------	|------	|------	|------	|------	|------	|------	|
| 5868 	| 25   	| 71   	| 55   	| 21   	| 0    	| 104  	| 0    	| 35   	| 0    	|
| 4    	| 5872 	| 3    	| 11   	| 2    	| 1    	| 6    	| 0    	| 4    	| 0    	|
| 25   	| 3    	| 5711 	| 8    	| 98   	| 1    	| 95   	| 0    	| 19   	| 0    	|
| 36   	| 77   	| 37   	| 5837 	| 47   	| 0    	| 49   	| 0    	| 15   	| 0    	|
| 19   	| 9    	| 129  	| 55   	| 5778 	| 0    	| 63   	| 0    	| 14   	| 0    	|
| 3    	| 1    	| 2    	| 4    	| 3    	| 5957 	| 1    	| 16   	| 7    	| 12   	|
| 24   	| 6    	| 27   	| 22   	| 39   	| 0    	| 5659 	| 0    	| 13   	| 0    	|
| 1    	| 1    	| 0    	| 1    	| 0    	| 26   	| 0    	| 5926 	| 12   	| 48   	|
| 19   	| 5    	| 20   	| 5    	| 12   	| 6    	| 21   	| 12   	| 5877 	| 0    	|
| 1    	| 1    	| 0    	| 2    	| 0    	| 9    	| 2    	| 46   	| 4    	| 5940 	|

```
Correct rate in training samples: 0.974
Correct rate in testing samples: 0.859
Overall hit rate: 0.957
Learning rate = 0.0500
EPOCHS = 500
```

Timed using time() on a 7th Gen i7, Ubuntu 18.04 machine the program execution time is shown below:
```
real	20m25,073s
user	154m41,118s
sys	    0m59,555s
```

---

- For **one pass of the training dataset  (1 epoch)**, using
the same parameters as before, the execution time, as well as the
overall hit rates are shown below for reference:


```	
    real	0m7,437s
    user	0m37,199s
    sys	    0m0,328s

    Correct rate in training samples: 0.773
    Correct rate in testing samples: 0.759
    Overall hit rate: 0.771
    Learning rate = 0.0500
    EPOCHS =  1
```