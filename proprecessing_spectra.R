library(MALDIquant)
library(MALDIquantForeign)

#' processSpectra
#'
#' @param input MALDIquant mass spectrum list
#'
#' @return MALDIquant mass spectrum list
#' @export
#'
processSpectra <- function(fileList, wd = 5, snr=3){
  # Intensity transformation
  spectra <- transformIntensity(input, method = "log") 
  # Baseline Correction
  spectra <- removeBaseline(spectra, method = "SNIP", iterations=100) 
  # Intensity Calibration / Normalization
  spectra <- calibrateIntensity(spectra, method="TIC")
  # Spectra alignment
  spectra <- alignSpectra(spectra, halfWindowSize = wd, allowNoMatches= TRUE,  tolerance = 0.002, warpingMethod="cubic")
  # Peaks detection
  peaks <- detectPeaks(spectra, method = "MAD", halfWindowSize = wd, SNR = snr)
}

# Create an intensity matrix
intensity_matrix <- intensityMatrix(peaks, spectra)

# write the output file
file_sp <- file(out_filename, open="wt")
write.table(intensity_matrix, file_sp, sep=" ", row.names=FALSE)
