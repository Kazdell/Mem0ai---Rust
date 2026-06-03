use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let out_dir = env::var("OUT_DIR").unwrap();
        let temp_dir = Path::new(&out_dir).join("rc_build");
        fs::create_dir_all(&temp_dir).unwrap();

        let temp_rc = temp_dir.join("resource.rc");
        let temp_ico = temp_dir.join("icon.ico");
        let temp_o = temp_dir.join("resource.o");
        let temp_a = temp_dir.join("libresource.a");

        // Copy icon.ico from project root to temp directory
        fs::copy("icon.ico", &temp_ico).unwrap();

        // Write resource.rc pointing to icon.ico in the same directory (numeric ID 1 is required for primary exe icon)
        fs::write(&temp_rc, "1 ICON \"icon.ico\"\n").unwrap();

        // Run windres in the temp directory where there are no spaces in paths
        let status_windres = Command::new("windres")
            .current_dir(&temp_dir)
            .arg("-i")
            .arg("resource.rc")
            .arg("-o")
            .arg("resource.o")
            .status();

        if status_windres.is_err() || !status_windres.unwrap().success() {
            panic!("Failed to compile resource file using windres. Please ensure windres is in your PATH.");
        }

        // Package resource.o into libresource.a using ar
        let status_ar = Command::new("ar")
            .current_dir(&temp_dir)
            .arg("rcs")
            .arg("libresource.a")
            .arg("resource.o")
            .status();

        if status_ar.is_err() || !status_ar.unwrap().success() {
            panic!("Failed to package resource object using ar. Please ensure ar is in your PATH.");
        }

        // Copy the compiled static library back to OUT_DIR
        let dest_a = Path::new(&out_dir).join("libresource.a");
        fs::copy(&temp_a, &dest_a).unwrap();

        // Tell Cargo to search in out_dir and force link the resource library
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg=-lresource");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        println!("cargo:rerun-if-changed=icon.ico");
        println!("cargo:rerun-if-changed=build.rs");

        // Cleanup temp directory files
        let _ = fs::remove_file(&temp_rc);
        let _ = fs::remove_file(&temp_ico);
        let _ = fs::remove_file(&temp_o);
        let _ = fs::remove_file(&temp_a);
        let _ = fs::remove_dir(&temp_dir);
    }
}


