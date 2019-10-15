def processing_plot():
    global freq_from, freq_to, this_array_has_a_plot, plot, graph_title, rot180, freq_step, angle_step, array, scale, grate, filters, filters_number
    x_corner=0
    y_corner=1200-790
    x_width= 1920
    y_height= 5
    freq_array=get_freq(rounded=0)
    subarray=array[y_corner:y_corner+y_height,x_corner:x_corner+x_width]
    mean_subarray= subarray.mean(axis=0)
    print (global_basename[:global_basename.find("_")], mean_subarray.max(), freq_array[mean_subarray.argmax()+x_corner])
    #plt.imsave(address_of_save_fig+'/'+global_basename.replace('dat','jpg'),  subarray)
    plt.imsave(address_of_save_fig+'/'+global_basename.replace('dat','png'),  array)
    f= open(address_of_save_fig+'/'+global_basename.replace('dat','txt'), "a")
    #np.savetxt(f, mean_subarray, fmt='%1.4f')
    f.close()
