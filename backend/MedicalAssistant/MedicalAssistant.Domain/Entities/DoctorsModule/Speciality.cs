namespace MedicalAssistant.Domain.Entities.DoctorsModule;

public class Specialty : BaseEntity
{
    public string Name { get; set; } = string.Empty;

    public ICollection<Doctor> Doctors { get; set; } = new HashSet<Doctor>();
}