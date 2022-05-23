#!/bin/bash
sbatch lxmert_mao-3_normonly-expanded-nc-pt0-avsc.sh
sbatch lxmert_mao-3_normonly-expanded-nc-pt4-avsc.sh
sbatch lxmert_mao-3_normonly-expanded-nc-pt7-avsc.sh
sbatch lxmert_mao-9_normonly-expanded-nc-pt0-avsc.sh
sbatch lxmert_mao-9_normonly-expanded-nc-pt4-avsc.sh
sbatch lxmert_mao-9_normonly-expanded-nc-pt7-avsc.sh

sbatch lxmert_mao-3_normonly-expanded-nc-pt0-avsc-scaled.sh
sbatch lxmert_mao-3_normonly-expanded-nc-pt4-avsc-scaled.sh
sbatch lxmert_mao-3_normonly-expanded-nc-pt7-avsc-scaled.sh
sbatch lxmert_mao-9_normonly-expanded-nc-pt0-avsc-scaled.sh
sbatch lxmert_mao-9_normonly-expanded-nc-pt4-avsc-scaled.sh
sbatch lxmert_mao-9_normonly-expanded-nc-pt7-avsc-scaled.sh
