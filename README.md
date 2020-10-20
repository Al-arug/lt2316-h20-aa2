## LT2316 H20 Assignment A2 : Ner Classification

Name: Ali Aruqi

## Notes on Part 1.

Lstm and Linnear function are used in the model. The size of the output of the lstm is reshaped from 3d to 2d then passed to the linear function. (num_samples,len_sample,num_features) (20,120,6) to (20*120,6).  

## Notes on Part 2.

No helper function used. The loss of the model is the accumilated loss of the loss for each batch of the last epoch only.

## Notes on Part 3.

my hyperparameters as seen in the run file are focused on pointing out the defference in preformance in optimizer and the learning rate. I was also curious to see if the batch size affects the training. I also had in mind that the number of layers should reasonably inhance the learning, so I included it in the hyperparamaters and my results in the next section showed that more layers does not necessarily decrease the loss. I was also interested in seeing how the drop rate affects the learning, in the case of my classifier- as mulit classes classifier- higher drop rate had a negative effect. The parallel helped by giving a comparative veiw which makes it easier to point the points of difference. 


## Notes on Part 4.

Adam with learning rate 0.001 was the best performance. Out of curiousity I'v tested all of the hyperparamaters comparitevly and got results as such: It was better performance when batch_size = 20 than batch_size = 10. Better with 2 layers than 4 layers or only one layer.  The lower the drop rate the better the performance was. 

## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*
