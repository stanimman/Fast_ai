# Forked from https://github.com/ernest-s/fastai_notebooks/blob/master/image_classification_from_csv.py

from utils import *
#from image_classification_from_folder import train_model

class DatasetFromCsv(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
                the first column of the csv file should have the file names
                the second column of the csv file should have the class names
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.img_col = self.labels.columns[0]
        self.cls_col = self.labels.columns[1]
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = self._find_classes()

    def __len__(self):
        return len(self.labels)
    
    def _find_classes(self):
        """
        Finds the classes in a dataset.
        Returns:
            class_to_idx: Dictionary mapping between classes and their
                            corresponding index
        """
        classes = list(self.labels[self.cls_col].unique())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                f"{self.labels.iloc[idx, 0]}.jpg")
        image = Image.open(img_name)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[self.labels.iloc[idx, 1]]
        return image, label

def predict_fn(model, path, device, num_classes, transform, file_path = None,
               TTA = False, center_weight = 1.0, mode = "test", 
               batch_size = 64, suffix = None):
    '''Implements prediction function for image classification
    Args:
        model: trained model
        path: system path to image files
        device: "cuda" or "cpu"
        num_classes: number of classes in dataset
        transform: transformation to be applied to the dataset
        file_path: system path to the csv file containing image names and 
                    classes (needed for "valid" mode)
        TTA: Whether test time augmentation needed
            TTA takes 4 corner crops and one center crop and averages the
            output for all 5 crops
        center_weight: Weight to be given to the center crop. By default
            it is 1 i.e. center crop is weighted equally as corner crops
        mode: "test" or "valid"
            for test, all images will be in one folder
            for valid, images will be in their respective class folders
        batch_size: batch size for prediction
        suffix: suffix if any to be added to the image name (e.g. ".jpg")
    Returns:
        results: Dictionary containing "img_names", "log_pred" for "test" mode
                    and "img_names", "log_pred", "val_y", "classes", for 
                    "valid" mode
            img_names: names of the images
            log_pred: log predictions
            val_y: True y
            classes: List of classes
    '''

    since = time()
    model = model.to(device)
    model.eval()
    
    results = {}
        
    if mode == "test":
        # If mode is 'test', then all the test images are in single folder
        num_examples = len(os.listdir(path))
        results["img_names"] = os.listdir(path)
    elif mode == "valid":
        # If mode is 'valid', then the images are in respective class folders
        results["val_y"] = []
        results["img_names"] = []
        label_file = pd.read_csv(file_path)
        img_col, cls_col = label_file.columns[:2]
        results["classes"] = label_file[cls_col].unique()
        results["classes"].sort()
        # create a dictionary of image names and their respective classes
        results["img_names"] = list(label_file[img_col])
        if suffix is not None:
            results["img_names"] = [i+suffix for i in results["img_names"]]
        val_y_ = list(label_file[cls_col])
        val_y_ = [np.where(results["classes"] == i)[0][0] for i in val_y_]
        results["val_y"] = val_y_
        num_examples = len(label_file)
    else:
        print("Invalid 'mode'. Please enter 'test' or 'valid'")
        return 0
    
    num_minibatch = int(np.ceil(num_examples / batch_size))
    
    mini_batch = 1
    
    run_log_pred = np.empty((0, num_classes), float)
    for i in range(num_minibatch):

        # Print status bar
        mini_batch_comp = int((mini_batch/num_minibatch)*100)//2
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" %("="*mini_batch_comp, 
                         2*mini_batch_comp))

        st_index = i * batch_size
        en_index = (i + 1) * batch_size
        img_names = results["img_names"][st_index:en_index]
        # Read image names if 'test' mode
        # Read image names + classes if 'valid' mode
        img_names = [f'{path}{j}' for j in img_names]
        if mode == "valid":
            cls_names = [results["classes"][j] for j in 
                         (results["val_y"][st_index:en_index])]

        images = None
        for j in img_names:
            # Read the images and apply appropriate transformations
            with open(j, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img = transform(img)
                img.unsqueeze_(0+TTA)
                if images is not None:
                    images = torch.cat((images, img), 0+TTA)
                else:
                    images = img.clone()
        
        if TTA == False:
            # Only one prediction is TTA is False
            if len(images.shape) != 4:
                print("Image shape is wrong. Check transforms")
            inputs = images.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            outputs = outputs.cpu().data.numpy()
            run_log_pred = np.append(run_log_pred, outputs, axis = 0)
        else:
            # Four corners + center crop prediction if TTA is True
            if len(images.shape) != 5:
                print("Image shape is wrong. Check transforms")
            cur_log_pred = np.zeros((images.shape[1], num_classes))
            for run in range(5):
                inputs = images[run,:,:,:,:]
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs = F.softmax(outputs, 1)
                outputs = outputs.cpu().data.numpy()
                if run == 4:
                    # Add weight factor for center crop prediction
                    cur_log_pred += (outputs * center_weight)
                else:
                    cur_log_pred += outputs
            cur_log_pred /= (4+center_weight)
            run_log_pred = np.append(run_log_pred, cur_log_pred, axis = 0)
        mini_batch += 1
    results["log_pred"] = run_log_pred
    elapsed = time() - since
    print("")
    print(f'Time taken for prediction {elapsed // 60}m {elapsed % 60}s')
        
    return results
    
def train_model(model, criterion, optimizer, dataloaders, device, 
          dataset_sizes, num_classes, scheduler = None, 
          num_epochs=25, metric=None, freeze_bn = False):
    '''Trains an image classification model
    Args:
        model: Network which has to be trained
        criterion: Lossfunction
        optimizer: pytorch optimizer function
        dataloaders: pytorch dataloader
        device: "cuda" or "cpu"
        dataset_sizes: Dictionary containing number of images in train and
                        valid
        num_classes: Number of classes in dataset
        scheduler: LR scheduler
        num_epochs: number of epochs
        metric: sklearn metric or any custom function that takes
                true value and predicted values as arguments and 
                outputs a number
        freeze_bn: Should batch norm layers be frozen? Set True while
                    using pre-trained models (big networks) for imagenet
                    like images
    Returns:
        model: trained model
        lr_hist: history of learning rate used in training
    '''
    lr_hist = []
    since = time()
    if freeze_bn == True:
        for mod in model.modules():
            if "BatchNorm" in mod.__class__.__name__:
                for p in mod.parameters():
                    p.requires_grad = False
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Count training and validation examples
    train_examples = len(dataloaders['train'].dataset)
    valid_examples = len(dataloaders['valid'].dataset)

    train_bs = dataloaders['train'].batch_size
    valid_bs = dataloaders['valid'].batch_size
    
    # Calculate number of minibatches for training and validation
    num_minibatch = {'train': int(np.ceil(train_examples / train_bs)), 
                     'valid': int(np.ceil(valid_examples / valid_bs))}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Track labels and predictions
        labels_ = np.zeros(0)
        preds_ = np.zeros(0)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            mini_batch = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                mini_batch += 1
                # Print status bar
                if phase == 'train':
                    mini_batch_comp = int((mini_batch/
                                           num_minibatch[phase])*100)//2
                    sys.stdout.write('\r')
                    sys.stdout.write("%s[%-50s] %d%%" %(phase, 
                                     "="*mini_batch_comp, 2*mini_batch_comp))
                else:
                    mini_batch_comp = int((mini_batch/
                                           num_minibatch[phase])*100)//5
                    sys.stdout.write('\r')
                    sys.stdout.write("%s[%-20s] %d%%" %(phase, 
                                     "="*mini_batch_comp, 5*mini_batch_comp))

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # Update schedule if scheduler is given
                        if scheduler is not None:
                            scheduler.step()
                            lr_ = []
                            for group in optimizer.param_groups:
                                lr_.append(group["lr"])
                            lr_hist.append(lr_)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # update labels and predictions
                labels_ = np.append(labels_, labels.cpu().numpy())
                preds_ = np.append(preds_, preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("")
            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # update metric value after every epoch
            if metric is not None:
                mvalue = metric(labels_, preds_)
                mname = metric.__name__
                print(f'{phase} {mname} : {mvalue}')

        print()

    time_elapsed = time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {best_acc}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, lr_hist
