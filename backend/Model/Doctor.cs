namespace backend.Models;

public class Doctor {
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string Specialty { get; set; } = string.Empty;
    public string Qualifications { get; set; } = string.Empty;
    public double Rating { get; set; }
    public int Reviews { get; set; }
    public decimal Fees { get; set; }
    public string Bio { get; set; } = string.Empty;
    public string HospitalId { get; set; } = string.Empty; 
}
