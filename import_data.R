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


# binning function for intensity matrix (e.g. csv format)
# min.mz : 1st column (usually equal to 1)
# max.mz : number of columns
# num.bin : number of bins
binner <- function(spectra, min.mz=min.mz, max.mz=max.mz, num.bin=num.bin){
  bin.width <-(max.mz - min.mz+1) / num.bin
  binned <- numeric(num.bin)
  for(i in 1:num.bin){
    binned[i] <- sum(spectra[(min.mz+(i-1)*bin.width):(i*bin.width)])
  }
  return(binned)
}
binned_sp_matrix <- t(apply(sp_matrix, 1, binner))
