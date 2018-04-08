from keras.layers import Permute, Reshape, Dense, Lambda, K, RepeatVector, merge


def attention_3d_block(inputs, time_steps: int, single_attention_vector: bool):
    # inputs.shape = (batch_size, time_steps, input_dim)
    print(inputs.shape)
    input_dim = int(inputs.shape[2])

    print(input_dim)
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul
