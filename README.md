# Compositional Neural Machine Translation based on the OpenNMT-py Toolkit

This software implements the Neural Machine Translation based on Compositional Source Word Representations as described in the paper: http://eamt2018.dlsi.ua.es/proceedings-eamt2018.pdf#page=51

### Compositional Source Word Embeddings
  To activate the source word representations from a character trigram vocabulary, select 
  
  ```-data_type text-trigram``` in the settings of preprocess.py 
  and 
  ```-model_type text-trigram``` during training and translation 


Previous features:
- [data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- [Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [TensorBoard/Crayon logging](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Image-to-text processing](http://opennmt.net/OpenNMT-py/im2text.html)
- [Speech-to-text processing](http://opennmt.net/OpenNMT-py/speech2text.html)

Beta Features (committed):
- multi-GPU
- ["Attention is all you need"](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- Structured attention
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper
- Inference time loss functions.

[Full Documentation](http://opennmt.net/OpenNMT-py/)


## Citation

If you use this software, please cite:

```
@inproceedings{eamt,
  author    = {Ataman Duygu and
               Mattia A. Di Gangi and
               Marcello Federico},
  title     = {Compositional Source Word Representations for Neural Machine Translation},
  booktitle = {Proceedings of the 21st Annual Conference of the European Association for Machine Translation},
  pages     = {31--40},
  year      = {2018}
}
```


[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
