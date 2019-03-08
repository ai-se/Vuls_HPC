#! /bin/tcsh
# chmod +x run_hpc.sh
rm err/*
rm out/*
#foreach FEA (text combine metrics random)
foreach FEA (semi)
  foreach VAR (`seq 0 1 29`)
    bsub -q standard -W 5000 -n 2 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share/tjmenzie/zyu9/miniconda2/bin/python2.7 runner.py error_hpcc_feature $FEA $VAR
  end
end

