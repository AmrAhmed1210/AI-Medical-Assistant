namespace backend.Models;

public class AdminStats {
    public int TotalUsers { get; set; }
    public int TotalDoctors { get; set; }
    public int TotalPatients { get; set; }
    public int TotalHospitals { get; set; }
    public long Revenue { get; set; }
    public int ActiveConsultations { get; set; }
}
