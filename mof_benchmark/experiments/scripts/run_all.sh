calc=$1

cd optimization
sbatch submit.sh $calc optimization.yaml
cd ../bulk_modulus
sbatch submit.sh $calc bulk_modulus.yaml
cd ../heat_capacity
sbatch submit.sh $calc heat_capacity.yaml
cd ../stability
sbatch submit_stability.sh $calc stability_prod_mtk.yaml
sbatch submit_temp.sh $calc stability_prod_temp_mtk.yaml
sbatch submit_copper.sh $calc stability_prod_copper_mtk.yaml
