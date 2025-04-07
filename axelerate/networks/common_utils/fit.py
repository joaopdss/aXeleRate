import shutil
import os
import time
import tensorflow as tf
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from axelerate.networks.yolo.backend.utils.map_evaluation import MapEvaluation
from axelerate.networks.common_utils.callbacks import WarmUpCosineDecayScheduler
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LambdaCallback
from datetime import datetime

metrics_dict = {'val_accuracy':['accuracy'],'val_loss':[],'mAP':[]}

def train(model,
		 loss_func,
		 train_batch_gen,
		 valid_batch_gen,
		 learning_rate = 1e-4,
		 nb_epoch = 300,
		 project_folder = 'project',
		 first_trainable_layer=None,
		 network=None,
		 metrics="val_loss",
		 custom_callback=None,
	 	 imgs_folder="data/dataset",
		 reduce_lr_config=None):
	"""A function that performs training on a general keras model.

	# Args
		model : keras.models.Model instance
		loss_func : function
			refer to https://keras.io/losses/

		train_batch_gen : keras.utils.Sequence instance
		valid_batch_gen : keras.utils.Sequence instance
		learning_rate : float
		saved_weights_name : str
	"""
	# Create project directory
	train_start = time.time()
	train_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	path = os.path.join(project_folder, train_date)
	basename = network.__class__.__name__ + "_best_"+ metrics
	print('Current training session folder is {}'.format(path))
	os.makedirs(path)
	save_weights_name = os.path.join(path, basename + '.h5')
	save_plot_name = os.path.join(path, basename + '.jpg')
	save_weights_name_ctrlc = os.path.join(path, basename + '_ctrlc.h5')
	print('\n')

	# 1 Freeze layers
	layer_names = [layer.name for layer in model.layers]
	fixed_layers = []
	if first_trainable_layer in layer_names:
		for layer in model.layers:
			if layer.name == first_trainable_layer:
				break
			layer.trainable = False
			fixed_layers.append(layer.name)
	elif not first_trainable_layer:
		pass
	else:
		print('First trainable layer specified in config file is not in the model. Did you mean one of these?')
		for i,layer in enumerate(model.layers):
			print(i,layer.name)
		raise Exception('First trainable layer specified in config file is not in the model')

	if fixed_layers != []:
		print("The following layers do not update weights!!!")
		print("	", fixed_layers)

	# 2 create optimizer
	optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	# 3. create loss function
	model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics_dict[metrics])
	model.summary()

	#4 create callbacks   
	
	tensorboard_callback = tf.keras.callbacks.TensorBoard("logs", histogram_freq=1)
	
	early_stop = EarlyStopping(monitor=metrics, 
					   min_delta=0.001, 
					   patience=20, 
					   mode='auto', 
					   verbose=1,
					   restore_best_weights=True)
					   
	checkpoint = ModelCheckpoint(save_weights_name, 
								 monitor=metrics, 
								 verbose=1, 
								 save_best_only=True, 
								 mode='auto', 
								 period=1)
								 
	reduce_lr = None
	if reduce_lr_config:
		# If configuration is provided, use those values
		print("Using custom ReduceLROnPlateau settings")
		reduce_lr = ReduceLROnPlateau(
			monitor=reduce_lr_config.get('monitor', metrics),
			factor=reduce_lr_config.get('factor', 0.2),
			patience=reduce_lr_config.get('patience', 10),
			min_lr=reduce_lr_config.get('min_lr', 0.00001),
			verbose=1
		)
	else:
		# Use default values
		reduce_lr = ReduceLROnPlateau(
			monitor=metrics, 
			factor=0.2,
			patience=10, 
			min_lr=0.00001,
			verbose=1
		)

	map_evaluator_cb = MapEvaluation(network, valid_batch_gen,
									 save_best=True,
									 save_name=save_weights_name,
									 iou_threshold=0.5,
									 score_threshold=0.3,
									 tensorboard=tensorboard_callback)

	warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate,
											total_steps=len(train_batch_gen)*nb_epoch,
											warmup_learning_rate=0.0,
											warmup_steps=len(train_batch_gen)*min(3, nb_epoch-1),
											hold_base_rate_steps=0,
											verbose=1)

	if network.__class__.__name__ == 'YOLO' and metrics =='mAP':
		callbacks = [tensorboard_callback, map_evaluator_cb, warm_up_lr]
	else:
		callbacks= [early_stop, checkpoint, warm_up_lr, tensorboard_callback] 
	
	if custom_callback is not None:
		print('adding custom_callback')
		callbacks.append(custom_callback)
	else:
		print('not using any custom_callbacks')
	
	# Add ReduceLROnPlateau to callbacks if it's enabled
	if reduce_lr:
		callbacks.append(reduce_lr)
	
	# class_weights = get_class_weights(train_batch_gen, imgs_folder)
	# print(f"Class weights: {class_weights}")
	# 4. training
	try:
		model.fit(train_batch_gen,
						steps_per_epoch  = len(train_batch_gen), 
						epochs		   = nb_epoch,
						validation_data  = valid_batch_gen,
						validation_steps = len(valid_batch_gen),
						callbacks		= callbacks,						
						verbose		  = 1,
						workers		  = 4,
						max_queue_size   = 10,
						use_multiprocessing = True)
	except KeyboardInterrupt:
		print("Saving model and copying logs")
		model.save(save_weights_name_ctrlc, overwrite=True, include_optimizer=False)
		shutil.copytree("logs", os.path.join(path, "logs"))  
		return model.layers, save_weights_name_ctrlc, None 
		
	shutil.copytree("logs", os.path.join(path, "logs"))
	_print_time(time.time()-train_start)
	
	# Generate correlation matrix plot if this is a classifier
	plot_path = None
	if network and network.__class__.__name__ == 'Classifier':
		try:
			plot_path = create_correlation_matrix_plot(model, valid_batch_gen, network._labels, project_folder)
		except Exception as e:
			print(f"Warning: Could not generate correlation matrix plot: {str(e)}")
	
	return model.layers, save_weights_name, plot_path

def _print_time(process_time):
	if process_time < 60:
		print("{:d}-seconds to train".format(int(process_time)))
	else:
		print("{:d}-mins to train".format(int(process_time/60)))

def get_class_weights(train_batch_gen, imgs_folder):
	class_weights = {}
	total_imgs = sum(len(files) for _, _, files in os.walk(imgs_folder))
	for key, val in train_batch_gen.class_indices.items():
		class_amount_imgs = len(os.listdir(os.path.join(imgs_folder, key)))
		class_weights[val] = (1 / class_amount_imgs) * (total_imgs / 2.0)
	
	return class_weights

def create_correlation_matrix_plot(model, validation_generator, labels, project_folder):
    """
    Creates a beautiful correlation matrix plot based on model predictions on validation data.
    
    Args:
        model: The trained Keras model
        validation_generator: Data generator for validation data
        labels: List of class labels
        project_folder: Folder to save the plot
        
    Returns:
        Path to the saved plot
    """
    print("\nGenerating correlation matrix and performance metrics from validation data...")
    
    # Get all validation data
    num_samples = validation_generator.samples
    steps = np.ceil(num_samples / validation_generator.batch_size)
    
    # Reset the generator to ensure we start from the beginning
    validation_generator.reset()
    
    # Get ground truth labels
    true_classes = validation_generator.classes
    
    # Get predictions
    predictions = model.predict(validation_generator, steps=steps, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
    cm = confusion_matrix(true_classes[:len(predicted_classes)], predicted_classes)
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes[:len(predicted_classes)], predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes[:len(predicted_classes)], 
                                                             predicted_classes, 
                                                             average='weighted')
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(20, 10))
    
    # Set overall style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # First subplot for confusion matrix
    ax1 = plt.subplot(1, 2, 1)
    
    # Custom colormap for better visibility
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Create beautiful heatmap with seaborn
    sns.heatmap(cm_normalized, annot=True, cmap=cmap, 
                xticklabels=labels, yticklabels=labels, 
                linewidths=0.2, fmt='.2f', annot_kws={"size": 10})
    
    # Make the plot more visually appealing
    plt.title("Normalized Confusion Matrix", fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Second subplot for metrics
    ax2 = plt.subplot(1, 2, 2)
    
    # Remove axis
    ax2.axis('off')
    
    # Create text for metrics
    metrics_text = (
        f"# Classification Performance Metrics\n\n"
        f"## Overall Metrics\n"
        f"- **Accuracy**: {accuracy:.4f}\n"
        f"- **Precision (weighted)**: {precision:.4f}\n"
        f"- **Recall (weighted)**: {recall:.4f}\n"
        f"- **F1 Score (weighted)**: {f1:.4f}\n\n"
        f"## Per-Class Metrics\n"
    )
    
    # Add per-class metrics
    report = classification_report(true_classes[:len(predicted_classes)], 
                                 predicted_classes, 
                                 target_names=labels, 
                                 output_dict=True)
    
    for label in labels:
        if label in report:
            metrics_text += (
                f"### {label}\n"
                f"- Precision: {report[label]['precision']:.4f}\n"
                f"- Recall: {report[label]['recall']:.4f}\n"
                f"- F1 Score: {report[label]['f1-score']:.4f}\n"
                f"- Support: {report[label]['support']}\n\n"
            )
    
    # Add the text to the plot
    ax2.text(0, 1, metrics_text, fontsize=12, va='top', 
             family='monospace', transform=ax2.transAxes)
    
    # Add title
    plt.suptitle('Classification Performance Analysis', fontsize=20, y=0.98)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.5, 0.01, f"Generated on {timestamp}", 
                ha="center", fontsize=10, style='italic')
    
    # Tight layout to ensure everything fits
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the detailed figure
    plot_path = os.path.join(project_folder, 'classification_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Also create a simpler correlation matrix for smaller display
    fig_small = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, cmap=cmap, 
                xticklabels=labels, yticklabels=labels, 
                linewidths=0.2, fmt='.2f')
    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Save the smaller figure
    small_plot_path = os.path.join(project_folder, 'confusion_matrix.png')
    plt.savefig(small_plot_path, dpi=200, bbox_inches='tight')
    
    # Check if we're in a Jupyter/IPython environment and display the plots
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display
            print("Displaying plots in notebook...")
            # Display the figures in the notebook
            display(fig)
            display(fig_small)
    except (ImportError, NameError):
        # Not in IPython environment
        pass
    
    plt.close('all')
    
    print(f"Performance analysis saved to: {plot_path}")
    
    # Save metrics as text file for reference
    metrics_file = os.path.join(project_folder, 'classification_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Classification Performance Metrics\n")
        f.write(f"================================\n\n")
        f.write(f"Generated on: {timestamp}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"- Accuracy: {accuracy:.4f}\n")
        f.write(f"- Precision (weighted): {precision:.4f}\n")
        f.write(f"- Recall (weighted): {recall:.4f}\n")
        f.write(f"- F1 Score (weighted): {f1:.4f}\n\n")
        f.write(f"Per-Class Metrics:\n")
        f.write(classification_report(true_classes[:len(predicted_classes)], 
                                     predicted_classes, 
                                     target_names=labels))
    
    return plot_path
