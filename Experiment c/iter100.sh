for (( i=1; i<=100; i++ ))
do
    python test.py --model_path model_trained/$model_name.pth --input_image_path image_test/LR_book.png --output_image_path result/F64B8_book_test.png --compare_image_path reference/HR_book.png --cuda
done
