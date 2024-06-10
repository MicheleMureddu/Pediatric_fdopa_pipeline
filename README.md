# Pediatric FDOPA Pipeline

## About

Pipeline to do analysis of pediatric F-DOPA scans.

## Installation
* Install [Python](https://www.python.org/downloads/)
* Install [pip](https://pip.pypa.io/en/stable/installation/)
* Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

`pip install --user nibabel antspyx matplotlib pandas` 

`git clone https://github.com/tfunck/pediatric_fdopa_pipeline`

## Useage
### To run pipeline with default settings:
python3 pediatric_fdopa_pipeline.py

### User options

* -i : Path for input data directory
* -o : Path for output file directory
* -s : Path for stereotaxic template file; default=atlas/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz
* -a : Path for stereotaxic label file; default='atlas/dka.nii.gz'
* --ref-labels : Label values to use for reference region with normal radioligand binding; default=[2]
* --roi-labels : Label values for target region (e.g., striatum); default=[11,12,13]

### Warning


The pipeline does not keep track of whether upstream files (e.g., *file_1*) are older than downstream files (e.g., *file_n*). Hence, if you delete an upstream file and rerun the pipline, it will not create a new version of the downstream file.

**pipeline_stage_1** --> *file_1* --> ...(more stages in pipeline)... --> **pipeline_stage_n** --> *file_n*

If you wish to re-run an analysis for a given set of subjects, it is recommended to delete the entire subject directories (or move them to a backup location) and re-run the entire pipeline for those subjects. This way there is no risk of ending up with out of date downstream files that do not reflect the new information in the upstream files that they depend on.

## Input Data formatting

All data must be formatted according to the [BIDS PET](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/09-positron-emission-tomography.html) format.


PET image file:

`<study_directory>/sub-<subject_id>/pet/sub-<subject_id>_pet.nii.gz`

PET header file:

`<study_directory>/sub-<subject_id>/pet/sub-<subject_id>_pet.json`

MRI image file:

`<study_directory>/sub-<subject_id>/anat/sub-<subject_id>_T1w.nii.gz`

### Example
`data/sub-1/pet/sub-1_pet.nii.gz`

`data/sub-1/pet/sub-1_pet.json`

`data/sub-1/anat/sub-1_T1w.nii.gz`

## PET .json header

Required fields:

* 'FrameDuration'
* 'FrameTimesStart'

### Example contents of a PET .json header :

{

	"FrameDuration": [
		60,
		60,
		60],
		
	"FrameTimesStart": [
		0,
		60,
		120]

}

## To Do

* Add QC images for identification and analysis

* Add TAC extraction, analysis, clustering

* Log parameters used to run analysis, both globally for entire pipeline and also for each processing stage

* Add more documentation

* Add testing for functions
