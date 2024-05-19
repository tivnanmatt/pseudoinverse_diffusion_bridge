import numpy as np
import torch
import nibabel as nib
import os
from torchvision import transforms
from PIL import Image
from lxml import etree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MICRONS_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, verbose=False):
        self.base_dir = base_dir
        self.active_filename = None
        self.active_data = None
        self.verbose = verbose
    def __len__(self):
        return 36864
    def __getitem__(self, indices):
        # indices might be a slice, if so, convert to a list
        if type(indices) == int:
            indices = np.array([indices])
        elif isinstance(indices, slice):
            _indices = []
            if indices.step is None:
                for i in range(indices.start, indices.stop):
                    _indices.append(i)
            else:
                for i in range(indices.start, indices.stop, indices.step):
                    _indices.append(i)
            indices = np.array(_indices)
        else:
            indices = np.array(indices)
        # break idx into subgroups that are in the same file
        indices_by_file_list = []
        file_list = []
        for iFile in range(36):
            indices_by_file_list.append(indices[indices//1024 == iFile])
            file_list.append(self.base_dir + '/sample_volume_%04d.npy' % iFile)
        # load the data
        X = []
        for iFile in range(36):
            # loading file 
            if len(indices_by_file_list[iFile]) == 0:
                continue
            elif self.active_filename == file_list[iFile]:
                X.append(self.active_data[indices_by_file_list[iFile] - iFile*1024])
            else:
                if self.verbose:
                    print('loading file %s' % file_list[iFile])
                self.active_filename = file_list[iFile]
                self.active_data = np.load(file_list[iFile]).reshape(-1,1,512,512)
                X.append(self.active_data[indices_by_file_list[iFile] - iFile*1024])
        
        X = np.concatenate(X, axis=0)
        X = torch.tensor(X, dtype=torch.float32)
        # need to reduce the last two dimensions from 512x512 to 256x256
        # do that by randomly sampling between 0 and 256 for both rows and cols
        # iRow = torch.randint(0, 254, (X.shape[0],))
        # iCol = torch.randint(0, 254, (X.shape[0],))
        # X = X[:,:,iRow:iRow+256,iCol:iCol+256]

        _X = torch.zeros((X.shape[0], 1, 256, 256))
        for i in range(X.shape[0]):
            iRow = torch.randint(0, 256, (1,))
            iCol = torch.randint(0, 256, (1,))
            _X[i] = X[i,:,iRow:iRow+256,iCol:iCol+256]
        X = _X
        return X
    

# brain patch dataset
class BrainPatchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, patch_shape, train_split=0.8, trainFlag=True, verbose=True, patchesPerImage=1):
        self.root_dir = root_dir
        all_brain_list = os.listdir(root_dir)
        # remove the element of self.brain_list that is 'overview'
        all_brain_list.remove('overview')
         # Calculate the split index
        split_idx = int(np.round(train_split * len(all_brain_list)))
        # Split the list into training and testing
        np.random.shuffle(all_brain_list)  # Shuffle the list
        
        if trainFlag:
            self.brain_list = all_brain_list[:split_idx]
        else:
            self.brain_list = all_brain_list[split_idx:]
        
        self.patch_shape = (64, 64, 64)

        self.patchesPerImage = patchesPerImage

        self.verbose = verbose

    def __len__(self):
        return len(self.brain_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]
        if type(idx) == slice:
            if idx.step is None:
                idx = range(idx.start, idx.stop)
            else:
                idx = range(idx.start, idx.stop, idx.step)
        
        # initialize the patch volume
        brain_patches = torch.zeros((len(idx), 3, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]))

        # load in the brain patches
        # for i, ind in enumerate(idx):
        #     if self.verbose:
        #         print('Loading brain patch %d/%d' % (i+1, len(idx)))
        #     brain_patches[i] = self._load_brain_patch(ind)

        for i, ind in enumerate(idx):
            if i % self.patchesPerImage == 0:
                print('Loading brain patch %d/%d' % (i+1, len(idx)))
                brain_patches[i:i+self.patchesPerImage] = self._load_brain_patches(ind)

        return brain_patches

    def _load_brain_patches(self, ind):

        #seed with ind
        np.random.seed(ind)
        _ind = np.random.randint(0, len(self.brain_list))

        # load in the brain
        brain_dir = os.path.join(self.root_dir, self.brain_list[_ind])
        mr_path = os.path.join(brain_dir, 'mr.nii.gz')
        ct_path = os.path.join(brain_dir, 'ct.nii.gz')
        mask_path = os.path.join(brain_dir, 'mask.nii.gz')
        
        mr = nib.load(mr_path).get_fdata()
        ct = nib.load(ct_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        mr_patches = torch.zeros((self.patchesPerImage, 1, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]))
        ct_patches = torch.zeros((self.patchesPerImage, 1, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]))
        mask_patches = torch.zeros((self.patchesPerImage, 1, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]))

        for iPatch in range(self.patchesPerImage):
        
            np.random.seed(2*ind*iPatch % 4294967296)
            _iSlice = np.random.randint(0, mr.shape[0] - self.patch_shape[0])
            np.random.seed(3*ind*iPatch % 4294967296)
            _iRow = np.random.randint(0, mr.shape[1] - self.patch_shape[1])
            np.random.seed(4*ind*iPatch % 4294967296)
            _iCol = np.random.randint(0, mr.shape[2] - self.patch_shape[2])

            # make it as a tensor with three channels
            mr_patches[iPatch] = torch.from_numpy(mr[_iSlice:_iSlice+self.patch_shape[0], _iRow:_iRow+self.patch_shape[1], _iCol:_iCol+self.patch_shape[2]])
            ct_patches[iPatch] = torch.from_numpy(ct[_iSlice:_iSlice+self.patch_shape[0], _iRow:_iRow+self.patch_shape[1], _iCol:_iCol+self.patch_shape[2]])
            mask_patches[iPatch] = torch.from_numpy(mask[_iSlice:_iSlice+self.patch_shape[0], _iRow:_iRow+self.patch_shape[1], _iCol:_iCol+self.patch_shape[2]])


        brain_patches = torch.cat((mr_patches, ct_patches, mask_patches), dim=1)

        return brain_patches
    



class SynthRad2023Task1CTSlice_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, verbose=False):
        self.root_dir = root_dir
        brain_list = os.listdir(root_dir)
        # remove the element of self.brain_list that is 'overview'
        brain_list.remove('overview')
        # Split the list into training and testing
        np.random.shuffle(brain_list)  # Shuffle the list 
        self.verbose = verbose
        self.brain_list = brain_list
        self.active_filename = None
        self.active_data = None
        self.top_of_head = None
        self.slices_per_volume = 60

    def __len__(self):
        return len(self.brain_list)*self.slices_per_volume
    
    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        elif type(idx) == slice:
            if idx.step is None:
                idx = range(idx.start, idx.stop)
            else:
                idx = range(idx.start, idx.stop, idx.step)
        
        # initialize the patch volume
        brain_ct_slices = np.zeros((len(idx), 256, 256)) - 1000

        # load in the brain patches
        for i, ind in enumerate(idx):
            iVolume = ind//self.slices_per_volume
            iSlice = ind%self.slices_per_volume
            # if self.verbose:
                # print('Loading brain volume %d/%d' % (iVolume+1, len(self.brain_list)))
            # brain_patches[i] = self._load_brain_patch(ind)
            brain_dir = os.path.join(self.root_dir, self.brain_list[iVolume])
            ct_path = os.path.join(brain_dir, 'ct.nii.gz')
            if self.active_filename != ct_path:
                if self.verbose:
                    print('loading file %s' % ct_path)
                ct_data = nib.load(ct_path).get_fdata()
                if ct_data.shape[0] > 256:
                    _iSlice = (ct_data.shape[0] - 256) // 2
                    ct_data = ct_data[_iSlice:_iSlice+256]
                if ct_data.shape[1] > 256:
                    _iRow = (ct_data.shape[1] - 256) // 2
                    ct_data = ct_data[:, _iRow:_iRow+256]
                if ct_data.shape[2] > 256:
                    _iCol = (ct_data.shape[2] - 256) // 2
                    ct_data = ct_data[:, :, _iCol:_iCol+256]
                # now handle the case where it is <256 by inserting it into a 256x256x256 volume on center
                _iSlice = (256 - ct_data.shape[0]) // 2
                _iRow = (256 - ct_data.shape[1]) // 2
                _iCol = (256 - ct_data.shape[2]) // 2
                _ct_data = np.zeros((256, 256, 256)) - 1000
                _ct_data[_iSlice:_iSlice+ct_data.shape[0], _iRow:_iRow+ct_data.shape[1], _iCol:_iCol+ct_data.shape[2]] = ct_data
                
                self.active_data = _ct_data
                self.active_filename = ct_path

            
                self.top_of_head = 0
                for ii in range(256):
                    if self.active_data[:,:,ii].mean() > -900:
                        self.top_of_head = ii

                # first 30 slices dont have much brain
                self.top_of_head -= 30

            brain_ct_slices[i] = self.active_data[:,:,self.top_of_head-iSlice]
            
    

        brain_ct_slices = torch.tensor(brain_ct_slices, dtype=torch.float32)
        
        # channel dimension
        brain_ct_slices = brain_ct_slices.unsqueeze(1)

        return brain_ct_slices
    





class SynthRad2023Task1MRISlice_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, verbose=False):
        self.root_dir = root_dir
        brain_list = os.listdir(root_dir)
        # remove the element of self.brain_list that is 'overview'
        brain_list.remove('overview')
        # Split the list into training and testing
        np.random.seed(2023)
        np.random.shuffle(brain_list)  # Shuffle the list 
        self.verbose = verbose
        self.brain_list = brain_list
        self.active_filename = None
        self.active_data = None
        self.top_of_head = None
        self.slices_per_volume = 60

    def __len__(self):
        return len(self.brain_list)*self.slices_per_volume
    
    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        elif type(idx) == slice:
            if idx.step is None:
                idx = range(idx.start, idx.stop)
            else:
                idx = range(idx.start, idx.stop, idx.step)
        
        # initialize the patch volume
        brain_mri_slices = np.zeros((len(idx), 256, 256))

        # load in the brain patches
        for i, ind in enumerate(idx):
            iVolume = ind // self.slices_per_volume
            iSlice = ind % self.slices_per_volume
            # if self.verbose:
                # print('Loading brain volume %d/%d' % (iVolume+1, len(self.brain_list)))
            # brain_patches[i] = self._load_brain_patch(ind)
            brain_dir = os.path.join(self.root_dir, self.brain_list[iVolume])
            mri_path = os.path.join(brain_dir, 'mr.nii.gz')
            if self.active_filename != mri_path:
                if self.verbose:
                    print('loading file %s' % mri_path)
                mri_data = nib.load(mri_path).get_fdata()
                if mri_data.shape[0] > 256:
                    _iSlice = (mri_data.shape[0] - 256) // 2
                    mri_data = mri_data[_iSlice:_iSlice+256]
                if mri_data.shape[1] > 256:
                    _iRow = (mri_data.shape[1] - 256) // 2
                    mri_data = mri_data[:, _iRow:_iRow+256]
                if mri_data.shape[2] > 256:
                    _iCol = (mri_data.shape[2] - 256) // 2
                    mri_data = mri_data[:, :, _iCol:_iCol+256]
                # now handle the case where it is <256 by inserting it into a 256x256x256 volume on center
                _iSlice = (256 - mri_data.shape[0]) // 2
                _iRow = (256 - mri_data.shape[1]) // 2
                _iCol = (256 - mri_data.shape[2]) // 2
                _mri_data = np.zeros((256, 256, 256))
                _mri_data[_iSlice:_iSlice+mri_data.shape[0], _iRow:_iRow+mri_data.shape[1], _iCol:_iCol+mri_data.shape[2]] = mri_data
            
                self.top_of_head = 0
                for ii in range(256):
                    if _mri_data[:,:,ii].mean() > 10:
                        self.top_of_head = ii

                # first 30 slices dont have much brain
                self.top_of_head -= 30

                self.active_data = _mri_data[:,:,self.top_of_head-self.slices_per_volume:self.top_of_head]
                self.active_data = self.active_data*128/np.mean(self.active_data[self.active_data>10])
                self.active_filename = mri_path

            brain_mri_slices[i] = self.active_data[:,:,iSlice]
            
        brain_mri_slices = torch.tensor(brain_mri_slices, dtype=torch.float32)
        brain_mri_slices.unsqueeze_(1)

        return brain_mri_slices





class ADNI_PET_Metadata:
    def __init__(self, filename):
        self.filename = filename
        self.project_identifier = None
        self.subject_identifier = None
        self.research_group = None
        self.subject_sex = None
        self.apoe_a1 = None
        self.apoe_a2 = None
        self.visit_identifier = None
        self.faq_total_score = None
        self.study_identifier = None
        self.subject_age = None
        self.weight_kg = None
        self.modality = None
        self.date_acquired = None
        self.imaging_description = None
        self.image_uid = None
        # Initialize each possible imaging protocol parameter explicitly
        self.manufacturer = None
        self.mfg_model = None
        self.radiopharmaceutical = None
        self.number_of_rows = None
        self.number_of_columns = None
        self.number_of_slices = None
        self.frames = None
        self.pixel_spacing_x = None
        self.pixel_spacing_y = None
        self.slice_thickness = None
        self.convolution_kernel = None
        self.counts_source = None
        self.randoms_correction = None
        self.attenuation_correction = None
        self.decay_correction = None
        self.reconstruction = None
        self.scatter_correction = None
        self.radioisotope = None
        self.load_metadata()

    def load_metadata(self):
        # Parse the XML file
        with open(self.filename, 'rb') as f:
            tree = etree.parse(f)
            root = tree.getroot()

        # Define namespace map based on your XML structure
        nsmap = root.nsmap
        default_ns = nsmap.get(None, '')

        # Fetch elements using XPath with proper namespace handling
        self.project_identifier = root.xpath('string(//projectIdentifier)', namespaces={'ns': default_ns})
        self.subject_identifier = root.xpath('string(//subjectIdentifier)', namespaces={'ns': default_ns})
        self.research_group = root.xpath('string(//researchGroup)', namespaces={'ns': default_ns})
        self.subject_sex = root.xpath('string(//subjectSex)', namespaces={'ns': default_ns})
        self.apoe_a1 = root.xpath('string(//subjectInfo[@item="APOE A1"])', namespaces={'ns': default_ns})
        self.apoe_a2 = root.xpath('string(//subjectInfo[@item="APOE A2"])', namespaces={'ns': default_ns})
        self.visit_identifier = root.xpath('string(//visitIdentifier)', namespaces={'ns': default_ns})
        self.faq_total_score = root.xpath('string(//component[@name="FAQ Total score"]/assessmentScore[@attribute="FAQTOTAL"])', namespaces={'ns': default_ns})
        self.study_identifier = root.xpath('string(//studyIdentifier)', namespaces={'ns': default_ns})
        self.subject_age = float(root.xpath('string(//subjectAge)', namespaces={'ns': default_ns}) or 0)
        self.weight_kg = float(root.xpath('string(//weightKg)', namespaces={'ns': default_ns}) or 0)
        self.modality = root.xpath('string(//modality)', namespaces={'ns': default_ns})
        self.date_acquired = root.xpath('string(//dateAcquired)', namespaces={'ns': default_ns})
        self.imaging_description = root.xpath('string(//description)', namespaces={'ns': default_ns})
        self.image_uid = root.xpath('string(//imageUID)', namespaces={'ns': default_ns})

        # Load imaging protocol parameters directly as attributes
        protocols = root.xpath('//protocolTerm/protocol', namespaces={'ns': default_ns})
        for protocol in protocols:
            term = protocol.get('term').replace(" ", "_").lower()  # Convert term to attribute-friendly format
            value = protocol.text
            if term and value:
                setattr(self, term, value)  # Dynamically set the attribute based on the term



        # now is the part where we find the path to the files
        # self.filename
        # get the base dir where this file is
        import os
        base_dir = os.path.dirname(self.filename)
        # first directory is the subject identifier
        subject_dir = os.path.join(base_dir, self.subject_identifier)
        # next directory is for the radiopharmaceutical
        date_dir = None
        image_dir = None
        imaging_description_dir = None

        # radiopharmaceutical_dir = os.path.join(subject_dir, self.imaging_description)
        # yes but replace spaces with underscores


        # if there is a space at the end, remove it
        imaging_description_dir = self.imaging_description
        while imaging_description_dir[-1] == " ":
            imaging_description_dir = imaging_description_dir[:-1]

        imaging_description_dir = os.path.join(subject_dir, imaging_description_dir.replace(" ", "_").replace(":", "_").replace("/", "_").replace("(", "_").replace(")", "_").replace(",", "_"))

        # assert that the directory exists
        if not os.path.exists(imaging_description_dir):
            raise Exception(f"Directory {imaging_description_dir} does not exist")
        
        # find what starts with the date
        for dir in os.listdir(imaging_description_dir):
            if dir.startswith(self.date_acquired):
                date_dir = os.path.join(imaging_description_dir, dir)
                break

        if date_dir is None:
            raise Exception(f"Could not find directory for date: {self.date_acquired}")
        
        if not os.path.exists(date_dir):
            raise Exception(f"Directory {date_dir} does not exist")
        
        # now get the path to the file
        for dir in os.listdir(date_dir):
            if self.image_uid in dir:
                image_dir = os.path.join(date_dir, dir)
                break

            # ADNI_029_S_4585_ADNI_Brain_PET__Raw_AV45_S931239_I1299449.xml

        if image_dir is None:
            # do a brute force search from the subject directory
            found_correct_flag = False
            for dir in os.listdir(subject_dir):
                for dir2 in os.listdir(os.path.join(subject_dir, dir)):
                    for dir3 in os.listdir(os.path.join(subject_dir, dir, dir2)):
                        if self.image_uid in dir3:
                            if found_correct_flag:
                                raise Exception(f"Had to resort to brute force, and there are multiple directories for subject: {self.subject_identifier} with the image uid: {self.image_uid}")
                            radiopharmaceutical_dir = os.path.join(subject_dir, dir)
                            date_dir = os.path.join(radiopharmaceutical_dir, dir2)
                            image_dir = os.path.join(date_dir, dir3)
                            found_correct_flag = True

        # if we still cant find it, then raise an exception
        if image_dir is None:
            raise Exception(f"Could not find directory for image uid: {self.image_uid}")
        

        # header files are the ones ending in .i.hdr
        self.header_files = []
        self.image_files = []
        for dir in os.listdir(image_dir):
            if dir.endswith('.i.hdr'):
                self.header_files.append(os.path.join(image_dir, dir))
        
        # now get the path to the image files by removing the .hdr
        if len(self.header_files) > 0:
            for header_file in self.header_files:
                self.image_files.append(header_file[:-4])
        else:
            for dir in os.listdir(image_dir):
                if dir.endswith('.v'):
                    self.image_files.append(os.path.join(image_dir, dir))
                    break
            if len(self.image_files) == 0:
                # raise Exception(f"No .hdr.i or .v files found in {image_dir}")
                # make this a warning instead
                print(f"No .hdr.i or .v files found in {image_dir}")

    def __str__(self):
        return "ADNI PET Metadata: \n" + \
               f"  Project Identifier: {self.project_identifier}\n" + \
               f"  Subject Identifier: {self.subject_identifier}\n" + \
               f"  Research Group: {self.research_group}\n" + \
               f"  Subject Sex: {self.subject_sex}\n" + \
               f"  APOE A1: {self.apoe_a1}\n" + \
               f"  APOE A2: {self.apoe_a2}\n" + \
               f"  Visit Identifier: {self.visit_identifier}\n" + \
               f"  FAQ Total Score: {self.faq_total_score}\n" + \
               f"  Study Identifier: {self.study_identifier}\n" + \
               f"  Subject Age: {self.subject_age}\n" + \
               f"  Weight (kg): {self.weight_kg}\n" + \
               f"  Modality: {self.modality}\n" + \
               f"  Date Acquired: {self.date_acquired}\n" + \
               f"  Imaging Description: {self.imaging_description}\n" + \
               f"  Image UID: {self.image_uid}\n" + \
               "  Imaging Protocol Parameters:\n" + \
               f"    Manufacturer: {self.manufacturer}\n" + \
               f"    Mfg Model: {self.mfg_model}\n" + \
               f"    Radiopharmaceutical: {self.radiopharmaceutical}\n" + \
               f"    Number of Rows: {self.number_of_rows}\n" + \
               f"    Number of Columns: {self.number_of_columns}\n" + \
               f"    Number of Slices: {self.number_of_slices}\n" + \
               f"    Frames: {self.frames}\n" + \
               f"    Pixel Spacing X: {self.pixel_spacing_x}\n" + \
               f"    Pixel Spacing Y: {self.pixel_spacing_y}\n" + \
               f"    Slice Thickness: {self.slice_thickness}\n" + \
               f"    Convolution Kernel: {self.convolution_kernel}\n" + \
               f"    Counts Source: {self.counts_source}\n" + \
               f"    Randoms Correction: {self.randoms_correction}\n" + \
               f"    Attenuation Correction: {self.attenuation_correction}\n" + \
               f"    Decay Correction: {self.decay_correction}\n" + \
               f"    Reconstruction: {self.reconstruction}\n" + \
               f"    Scatter Correction: {self.scatter_correction}\n" + \
               f"    Radioisotope: {self.radioisotope}\n" + \
              f"  Header Files: {self.header_files}\n" + \
              f"  Image Files: {self.image_files}\n" 



class ADNIPETSlice_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, verbose=False):
        self.root_dir = root_dir
        self.verbose = verbose
        self.metadata_filename_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.xml')]
        self.metadata_list_all = [ADNI_PET_Metadata(f) for f in self.metadata_filename_list]
        self.metadata_list = []  # Store tuples of (metadata, image_file_index)
        self.cumulative_slice_indices = []  # Cumulative count of slices per volume

        cumulative_slices = 0
        for meta in self.metadata_list_all:
            if meta.manufacturer == "Siemens ECAT" and meta.image_files is not None and meta.radiopharmaceutical == '18F-FDG':
                for i in range(len(meta.image_files)):
                    self.metadata_list.append((meta, i))
                    cumulative_slices += int(float(meta.number_of_slices))
                    self.cumulative_slice_indices.append(cumulative_slices)

        self.active_volume = None
        self.active_volume_index = -1
        self.active_data = None

    def __len__(self):
        return self.cumulative_slice_indices[-1] if self.cumulative_slice_indices else 0

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        elif type(idx) == slice:
            if idx.step is None:
                idx = range(idx.start, idx.stop)
            else:
                idx = range(idx.start, idx.stop, idx.step)

        slices = []

        for i in idx:
            volume_idx, slice_idx_within_volume = self._find_volume_and_slice_index(i)
            if volume_idx != self.active_volume_index:
                # Load new volume if necessary
                self.active_volume_index = volume_idx
                self.active_data = self._load_ADNI_volume(*self.metadata_list[volume_idx])
                if self.verbose:
                    print(f"Loaded volume {volume_idx} from file {self.metadata_list[volume_idx][0].image_files[self.metadata_list[volume_idx][1]]}")

            # Fetch the slice
            slice_data = self.active_data[slice_idx_within_volume, :, :]
            norm = np.max(slice_data)
            norm = np.minimum(norm, 1.0)
            slice_data = slice_data / norm
            slices.append(slice_data)

        slices = torch.tensor(slices, dtype=torch.float32).unsqueeze(1)  # Adding channel dimension
        return slices

    def _load_ADNI_volume(self, metadata, volume_index):
        filename = metadata.image_files[volume_index]
        dtype = np.float32  # Assuming 4 bytes per voxel as mentioned
        number_of_rows = int(float(metadata.number_of_rows))
        number_of_columns = int(float(metadata.number_of_columns))
        number_of_slices = int(float(metadata.number_of_slices))
        num_voxels = number_of_rows * number_of_columns * number_of_slices
        with open(filename, 'rb') as file:
            volume_data = np.fromfile(file, dtype=dtype, count=num_voxels)
            volume_data = volume_data.reshape((number_of_slices, number_of_rows, number_of_columns))
        return volume_data

    def _find_volume_and_slice_index(self, slice_idx):
        # Find which volume and which slice index within that volume corresponds to the global slice index
        for i, cumulative_slices in enumerate(self.cumulative_slice_indices):
            if slice_idx < cumulative_slices:
                volume_idx = i
                slice_idx_within_volume = slice_idx if i == 0 else slice_idx - self.cumulative_slice_indices[i - 1]
                return volume_idx, slice_idx_within_volume
        raise ValueError(f"Invalid slice index {slice_idx}")
    

class CelebA_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, verbose=False):
        """
        Args:
            image_dir (string): Directory with all the images.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            # Convert image to tensor
            transforms.ToTensor(),
            # Normalize with mean and std of CelebA dataset or ImageNet if unknown.
            # These values are placeholders; adjust them if necessary.
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # crop away the first and last column and first and last row
            # to go from 218x178 to 216x176
            # transforms.CenterCrop((216, 176))
            # actually, lets instead pad it to by 256x256
            transforms.Pad(((256-178)//2, (256-218)//2), fill=0)
        ])
        
        self.verbose = verbose
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, indices):
        # indices might be a slice, if so, convert to a list
        if type(indices) == int:
            indices = np.array([indices])
        elif isinstance(indices, slice):
            _indices = []
            if indices.step is None:
                for i in range(indices.start, indices.stop):
                    _indices.append(i)
            else:
                for i in range(indices.start, indices.stop, indices.step):
                    _indices.append(i)
            indices = np.array(_indices)
        else:
            indices = np.array(indices)
        # load the data
        X = []
        for i in indices:
            # loading file
            if self.verbose:
                print('loading file %s' % self.image_files[i]) 
            img_path = os.path.join(self.image_dir, self.image_files[i])
            img = Image.open(img_path)
            img = self.transform(img)
            X.append(img)
        
        X = torch.stack(X, dim=0)
        assert X.shape[1] == 3
        assert X.shape[2] == 256
        assert X.shape[3] == 256
        return X






# ../../data/chest_xray/train/COVID19/
# there are some covid chest x-ray images in .jpg format in that directory, lets make a dataset
    

class CovidChestXRay_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, verbose=False):
        """
        Args:
            image_dir (string): Directory with all the images.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            # Convert image to tensor
            transforms.ToTensor(),
            # Normalize with mean and std of CelebA dataset or ImageNet if unknown.
            # These values are placeholders; adjust them if necessary.
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.verbose = verbose
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, indices):
        # indices might be a slice, if so, convert to a list
        if type(indices) == int:
            indices = np.array([indices])
        elif isinstance(indices, slice):
            _indices = []
            if indices.step is None:
                for i in range(indices.start, indices.stop):
                    _indices.append(i)
            else:
                for i in range(indices.start, indices.stop, indices.step):
                    _indices.append(i)
            indices = np.array(_indices)
        else:
            indices = np.array(indices)
        # load the data
        X = []
        for i in indices:
            # loading file
            if self.verbose:
                print('loading file %s' % self.image_files[i]) 
            img_path = os.path.join(self.image_dir, self.image_files[i])
            img = Image.open(img_path)
            img = self.transform(img)
            X.append(img)



        # they may be different sizes so lets pad them all to 1024x1024
            
        _X = []
        for i in range(len(X)):
            img = X[i]

            if img.shape[0] == 4:
                img = img[:3]

            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            img = torch.mean(img, dim=0, keepdim=True)

            img = transforms.Resize((1024, 1024))(img)

            _X.append(img)
        X = _X

        X = torch.stack(X, dim=0)
        return X






        
if __name__ == '__main__':
    # my_MICRONS_Dataset = MICRONS_Dataset('../20231104_microns/data/microns_samples', verbose=True)
    # for iStart in range(0, 5000, 700):
    #     X = my_MICRONS_Dataset[iStart:iStart+700]
    #     print('MICRONS_Dataset[%d:%d]' % (iStart, iStart+700))
    #     print('     mean: %f' % X.mean().item())
    #     print('     std: %f' % X.std().item())
    #     print('     shape: %s' % str(X.shape))


    # my_SynthRad2023Task1CTSlice_Dataset = SynthRad2023Task1CTSlice_Dataset('../../data/SynthRad2023/Task1/brain/', verbose=True)

    # for iStart in range(0, 700, 70):
    #     X = my_SynthRad2023Task1CTSlice_Dataset[iStart:iStart+70]
    #     print('SynthRad2023Task1CTSlice_Dataset[%d:%d]' % (iStart, iStart+70))
    #     print('     mean: %f' % X.mean().item())
    #     print('     std: %f' % X.std().item())
    #     print('     shape: %s' % str(X.shape))


    # my_SynthRad2023Task1MRISlice_Dataset = SynthRad2023Task1MRISlice_Dataset('../../data/SynthRad2023/Task1/brain/', verbose=True)

    # for iStart in range(0, 5000, 700):
    #     X = my_SynthRad2023Task1MRISlice_Dataset[iStart:iStart+700]
    #     print('my_SynthRad2023Task1MRISlice_Dataset[%d:%d]' % (iStart, iStart+700))
    #     print('     mean: %f' % X.mean().item())
    #     print('     std: %f' % X.std().item())
    #     print('     shape: %s' % str(X.shape))




    # my_SynthRad2023Task1CTSlice_Dataset = SynthRad2023Task1CTSlice_Dataset('../../data/SynthRad2023/Task1/brain/', verbose=True)

    # X = my_SynthRad2023Task1CTSlice_Dataset[0:1000]
    
    # from matplotlib import pyplot as plt
    # from matplotlib import animation

    # fig = plt.figure()
    # im = plt.imshow(X[0], cmap='gray', vmin=500, vmax=1500)
    # plt.colorbar()
    
    # def updatefig(i):
    #     print('updatefig(%d)' % i)
    #     im.set_array(X[i*5])
    #     return im,

    # ani = animation.FuncAnimation(fig, updatefig, frames=range(200), blit=True)
    # writer = animation.writers['ffmpeg'](fps=10)
    # ani.save('ct_bone_window.mp4', writer=writer)
    # plt.close('all')

    # fig = plt.figure()
    # im = plt.imshow(X[0], cmap='gray', vmin=20, vmax=50)
    # plt.colorbar()
    
    # def updatefig(i):
    #     print('updatefig(%d)' % i)
    #     im.set_array(X[i*5])
    #     return im,

    # ani = animation.FuncAnimation(fig, updatefig, frames=range(200), blit=True)
    # writer = animation.writers['ffmpeg'](fps=10)
    # ani.save('ct_brain_window.mp4', writer=writer)
    # plt.close('all')

    # my_SynthRad2023Task1MRISlice_Dataset = SynthRad2023Task1MRISlice_Dataset('../../data/SynthRad2023/Task1/brain/', verbose=True)

    # X = my_SynthRad2023Task1MRISlice_Dataset[0:1000]
    
    # from matplotlib import pyplot as plt
    # from matplotlib import animation

    # fig = plt.figure()
    # im = plt.imshow(X[0], cmap='gray', vmin=0, vmax=X.max())
    # plt.colorbar()
    
    # def updatefig(i):
    #     print('updatefig(%d)' % i)
    #     im.set_array(X[i*5])
    #     return im,

    # ani = animation.FuncAnimation(fig, updatefig, frames=range(200), blit=True)
    # writer = animation.writers['ffmpeg'](fps=10)
    # ani.save('mri.mp4', writer=writer)

    my_CelebA_Dataset = CelebA_Dataset('../../data/celeba_aligned/img_align_celeba/', verbose=True)

    X = my_CelebA_Dataset[0:200]

    from matplotlib import pyplot as plt
    from matplotlib import animation
    # wide figure
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(X[0].permute(1,2,0)[::8,::8,:])
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(X[0].permute(1,2,0)[::8,::8,:])
    # plt.colorbar()
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(X[0].permute(1,2,0))

    X_down = 0*X
    # X_down = torch.conv2d(X, torch.ones(1,1,3,3)/9, stride=1)/torch.conv2d(torch.ones_like(X), torch.ones(1,1,3,3)/9, stride=1)
    X_down[:,0:1] = torch.conv2d(X[:,0:1], torch.ones(1,1,3,3)/9, stride=1, padding=1)/torch.conv2d(torch.ones_like(X[:,0:1]), torch.ones(1,1,3,3)/9, stride=1, padding=1)
    X_down[:,1:2] = torch.conv2d(X[:,1:2], torch.ones(1,1,3,3)/9, stride=1, padding=1)/torch.conv2d(torch.ones_like(X[:,1:2]), torch.ones(1,1,3,3)/9, stride=1, padding=1)
    X_down[:,2:3] = torch.conv2d(X[:,2:3], torch.ones(1,1,3,3)/9, stride=1, padding=1)/torch.conv2d(torch.ones_like(X[:,2:3]), torch.ones(1,1,3,3)/9, stride=1, padding=1)
    X_down = X_down[:, :, ::8, ::8]
    X_down = torch.clamp(X_down, 0, 1)
    X_down_and_noise = X_down + torch.randn_like(X_down)*0.1
    X_down_and_noise = torch.clamp(X_down_and_noise, 0, 1)

    def updatefig(i):
        print('updatefig(%d)' % i)
        im1.set_array(X_down[i].permute(1,2,0))
        im2.set_array(X_down_and_noise[i].permute(1,2,0))
        im3.set_array(X[i].permute(1,2,0))
        return im1,im2,im3

    ani = animation.FuncAnimation(fig, updatefig, frames=range(20), blit=True)

    writer = animation.writers['ffmpeg'](fps=10)

    ani.save('celeba.mp4', writer=writer)




    # my_CovidChestXRay_Dataset = CovidChestXRay_Dataset('../../data/chest_xray/train/COVID19/', verbose=True)

    # X = my_CovidChestXRay_Dataset[0:200]

    # from matplotlib import pyplot as plt
    # from matplotlib import animation

    # fig = plt.figure()
    # im = plt.imshow(X[0].permute(1,2,0))
    # plt.colorbar()

    # def updatefig(i):
    #     print('updatefig(%d)' % i)
    #     im.set_array(X[i].permute(1,2,0))
    #     return im,

    # ani = animation.FuncAnimation(fig, updatefig, frames=range(200), blit=True)

    # writer = animation.writers['ffmpeg'](fps=10)

    # ani.save('covid_chest_xray.mp4', writer=writer)