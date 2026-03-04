namespace backend.Models;

public class Hospital {
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string Address { get; set; } = string.Empty;
    public string Phone { get; set; } = string.Empty;
    public double Rating { get; set; }
    public int Reviews { get; set; }
    public string Image { get; set; } = string.Empty;
    public string Status { get; set; } = "active"; // active | inactive
    
    // الربط مع الأطباء (Relationship)
    public List<Doctor> Doctors { get; set; } = new();
}
