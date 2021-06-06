$startTime = Get-Date;
Write-Output "starting at: $($startTime.TimeOfDay)";



# Make an array/list of values so we can foreach on it
$list = 1200..16000;

$Job = $list | ForEach-Object -Parallel { 
    $outpath = "images/" + $_ + ".jpg";
    if(!(Test-Path $outpath)){
        $url = "http://thecoverproject.net/download_cover.php?src=cdn&amp;cover_id=" + $_;
        Invoke-WebRequest -Uri $url -OutFile $outpath;
        Write-Host "Got $_"
    }
} -AsJob

$job | Wait-Job | Receive-Job

$endTime = Get-Date;

Write-Output "Ended at: $($endTime.TimeOfDay)";
Write-Output "Elapsed time: $($endTime.Subtract($startTime).TotalSeconds) seconds.";