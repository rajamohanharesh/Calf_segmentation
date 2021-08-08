


def train(train_df,val_df,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,BATCH_SIZE,NUM_EPOCHS,modelPath):
	train_X,train_y,train_info = load_data(train_df,val_df,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,train_mode=1)
	val_X,val_y,val_info = load_data(train_df,val_df,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,train_mode=0)

	model = unet()
    mc = ModelCheckpoint(modelPath, monitor='val_dice_multi', mode='min', verbose=1, save_best_only=True)

    model.compile(optimizer = Adam(lr = LEARNING_RATE), loss = full_cost, metrics = ['accuracy', youden, dice_multi, cross_entropy])
    model.summary()
    history = model.fit(train_X,
              train_y,
              validation_data = (val_X,val_y),
              batch_size = BATCH_SIZE,
              epochs = NUM_EPOCHS,
              callbacks=[mc],shuffle=True)

    return history.history["loss"],history.history["val_loss"],history.history["dice_multi"],history.history["val_dice_multi"]


def evaluate(val_df,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,BATCH_SIZE,NUM_EPOCHS,modelPath,metrics):

	val_X,val_y,val_info = load_data(train_df,val_df,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,train_mode=0)

	val_n_slices = val_X.shape[0]

	predictions = model.predict(val_X)
    pred_categ, actual = flat_categ(predictions, val_y)

    for metric in metrics:
    	save_df()

	model = unet()
    mc = ModelCheckpoint(modelPath, monitor='val_dice_multi', mode='min', verbose=1, save_best_only=True)

    model.compile(optimizer = Adam(lr = LEARNING_RATE), loss = full_cost, metrics = ['accuracy', youden, dice_multi, cross_entropy])
    model.summary()
    model.fit(train_X,
              train_y,
              validation_data = (val_X,val_y),
              batch_size = BATCH_SIZE,
              epochs = NUM_EPOCHS,
              callbacks=[mc])