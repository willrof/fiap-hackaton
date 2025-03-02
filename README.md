Notas:
Devido ao tamanho dos arquivos de dataset, não estão incluídos aqui.

O arquivo `0 - data_cleaning.ipynb` considera que na mesma pasta de onde ele é executado:
- O arquivo `validacao.csv` está na pasta raiz.
- Os arquivos csv de treino estão numa subpasta chamada `training`.
- Os arquivos csv de itens estão em uma subpasta chamada `itens`.

O arquivo `1 - train_model.py` considera que os arquivos `.parquet` estão nesma pasta que ele.

O arquivo `2 - predict_api.py` considera que o modelo salvo está numa subpasta chamada `saved_model`.
