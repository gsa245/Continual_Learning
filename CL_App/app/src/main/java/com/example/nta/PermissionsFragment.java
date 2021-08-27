package com.example.nta;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Process;
import android.widget.Toast;

import androidx.fragment.app.Fragment;

/**
 * The sole purpose of this fragment is to request the necessary permissions.
 * It does not create a view.
 */
public class PermissionsFragment extends Fragment {

    private static final int PERMISSIONS_REQUEST_CODE = 10;
    private static final String[] PERMISSIONS_REQUIRED = {Manifest.permission.CAMERA};

    private PermissionsAcquiredListener callback;

    public void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);

        if (!hasPermissions()){
            requestPermissions(PERMISSIONS_REQUIRED,PERMISSIONS_REQUEST_CODE);
        }else{
            callback.onPermissionsAcquired();
        }
    }

    private boolean hasPermissions(){
        for(String permission: PERMISSIONS_REQUIRED){
            if(getContext().checkPermission(permission, Process.myPid(), Process.myUid()) !=
            PackageManager.PERMISSION_GRANTED){
                return  false;
            }
        }
        return true;
    }


    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(getContext(), "Camera permission granted", Toast.LENGTH_LONG).show();
                callback.onPermissionsAcquired();
            } else {
                Toast.makeText(getContext(), "Camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    public void setOnPermissionsAcquiredListener(PermissionsAcquiredListener callback) {
        this.callback = callback;
    }

    /**
     * Should be implemented by the host activity to get notified about permissions being acquired.
     */
    public interface PermissionsAcquiredListener {
        void onPermissionsAcquired();
    }

}
