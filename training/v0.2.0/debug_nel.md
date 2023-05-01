<!-- debug message for NEL compoent -->

So I am trying to add an entity linker to a Danish pipeline (see project [here](https://github.com/centre-for-humanities-computing/DaCy/tree/training-v0.2.0/training/v0.2.0)). 
It currently 

https://wandb.ai/kenevoldsen/dacy-v0.2.0?workspace=user-kenevoldsen
And have been trying

Debug steps:
- Does it train with a simple pipeline?

Using da_core_news_sm we obtain a NEL_MICRO_F score of ~34
Q: that is what is starts out with as well... will it improve?
--> we should try to train for longer 

Q: How does the loss look like?
--> I will have to log to wandb

Q: How does it look like for a da_core_news_lg pipeline?
--> currently running
A: 34.57 (start out that way as well) but keeping stable?