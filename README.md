# Natural-Language-To-Bash-Commands-Translation


This project is aimed at exploring different methods to convert a given natural language description (a given requirement in natural language) to a single line bash command. 


## Dependencies
1. pytorch==1.7.1
2. [bashlint](https://github.com/IBM/clai/tree/nlc2cmd/utils/bashlint)
3. spacy

## Data
The train/valid/test splits of the dataset can be found in the 'data' folder. <br />


|Data Split |Size|
|----|-----|
|train|25159|
|dev|5159|
|test|5159|

## Models

1. Seq2Seq uses a simple RNN based model to implement the above problem.
2. Seq2Seq_Attention uses Bahdanau attention along with the RNN based model to achieve better performance.
3. Seq2Seq_Attention_ut has an added auxiliary part to predict the utilities which should be present in a command and help the decode produce better results. The model is explained through a disgram in 'model.pdf'.
4. Seq2Seq_Transformers has a transformer encoder decoder architecture to achieve the above job. It gives the best result among all the methods.

## References

1. [Nl2Cmd Competition](http://nlc2cmd.us-east.mybluemix.net/#/participate)
2. [Project CLAI](https://arxiv.org/pdf/2002.00762.pdf)
