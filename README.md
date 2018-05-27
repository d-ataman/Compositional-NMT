# Compositional Neural Machine Translation based on the OpenNMT-py Toolkit


This software implements the Neural Machine Translation based on Compositional Source Word Representations as described in the paper: http://eamt2018.dlsi.ua.es/proceedings-eamt2018.pdf#page=51

## Compositional Source Word Embeddings

  To activate the source word representations from a character trigram vocabulary, select 
  
  ```-data_type text-trigram``` in the settings of preprocess.py 
  and 
  ```-model_type text-trigram``` during training and translation 


## Further information

For information about how to install and use OpenNMT-py:
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
