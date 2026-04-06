using MedicalAssistant.Domain.Entities.DoctorsModule;

namespace MedicalAssistant.Domain.Entities.DoctorsModule;

public class Specialty : BaseEntity
{
    public string Name { get; set; } = string.Empty;

    public string? Description { get; set; }

    public ICollection<Doctor> Doctors { get; set; } = new HashSet<Doctor>();
}
