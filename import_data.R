library(MSnbase)

# Create an intensity matrix from a list of MS spectra
# spectra_files must be in mzXML format

import_sp <- function(spectra_files) {
  # Use spectra only with MS level 1
  sp_data <- readMSData(spectra_files, msLevel=1)
  # TIC (total ion count) > 1e4
  sp_data <- sp_data[tic(sp_data)> 1e4]
  # Bin the intensities values according to the bin size
  bined_sp <- do.call(rbind, MSnbase::intensity(bin(sp_data, binSize=0.1)))
}
