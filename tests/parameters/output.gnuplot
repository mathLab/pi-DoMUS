plot 'output.gpl' using 1:2 w l title 'exp', \
     'output.gpl' using 1:(exp(-$1)) w l title 'exact'
