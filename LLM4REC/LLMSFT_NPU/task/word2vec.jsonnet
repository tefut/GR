{
  name: "word2vec_model_train",
  tasks: [
    {
      name: "train_word2vec",
      type: "train_word2vec",
      parameters: {
        corpus: {
          type: "line_sentence",
          source: "$data_dir/recall_data/positive_song_seq_data/positive_song_seq.csv"
        },
        save_model_path: "$data_dir/recall_data/word2vec/word2vec_new.model",
        save_embedding_path: "$data_dir/recall_data/word2vec/song_embedding.csv",
        embedding_file_config: {
          type: "txt"
        },
        vector_size: 64
      },
      outputs: "model"
    }
  ]
}
