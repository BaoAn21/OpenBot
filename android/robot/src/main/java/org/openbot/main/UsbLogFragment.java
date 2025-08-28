package org.openbot.main;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import org.openbot.R;
import org.openbot.databinding.FragmentUsbLogBinding;

public class UsbLogFragment extends Fragment {

    private MainViewModel mViewModel;
    private FragmentUsbLogBinding binding;
    private TextView logText;
    private boolean isPaused = false;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentUsbLogBinding.inflate(inflater, container, false);
        logText = binding.logText;
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        mViewModel = new ViewModelProvider(requireActivity()).get(MainViewModel.class);
        mViewModel.getLogMessages().observe(getViewLifecycleOwner(), messages -> {
            if (!isPaused) {
                logText.setText("");
                for (String message : messages) {
                    logText.append(message + "\n");
                }
            }
        });

        binding.clearButton.setOnClickListener(v -> {
            mViewModel.clearLogs();
        });

        binding.pauseButton.setOnClickListener(v -> {
            isPaused = !isPaused;
            if (isPaused) {
                binding.pauseButton.setImageResource(R.drawable.ic_play_arrow);
            } else {
                binding.pauseButton.setImageResource(R.drawable.ic_pause);
                // Update the log view with the latest messages
                logText.setText("");
                for (String message : mViewModel.getLogMessages().getValue()) {
                    logText.append(message + "\n");
                }
            }
        });
    }
}

