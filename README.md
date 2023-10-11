# Why to build ?

- Kiwix team need a GPT like platform to be available to use in remote areas

- It should be trained on custom data (flat files / zim files)

- Knowledge / Tokenized data should be available offline & transferrable from one system to another

- No GPU should be required for using the GPT (Only CPU - Mid size)

# What to build ?

- Build a GPT like platform which can perform Q/A offline

- It should be trained on specific data (It can extend already existing knowledge)

- Training can be on Flat Files / Zim Files / Json (Q/K/V Pattern) etc

- Should be standalone deployed on small - mid level system to get results

# How to build ?

- Source ticket(https://github.com/kiwix/overview/issues/93)

1. Use pre-built Model for tokenizer
2. Read flat file
3. Tokenize data
4. Create config for MLM / NSP model for Training
5. Train
6. Store Knowledge / Tokens in Flat files (easy to share)
7. Test

# Resources

https://saturncloud.io/blog/training-a-bert-model-from-scratch-with-hugging-face-a-comprehensive-guide/

https://huggingface.co/datasets/yelp_review_full

https://huggingface.co/docs/transformers/training
